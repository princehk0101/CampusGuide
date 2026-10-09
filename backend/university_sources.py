from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Literal
from urllib.parse import urljoin, urlparse

import requests
import pymupdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from backend.config import PROJECT_ROOT


UNIVERSITY_DATA_DIR = PROJECT_ROOT / "data" / "bundelkhand_university"
MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_TEXT_CHARACTERS = 500_000
MAX_LINKED_SOURCE_COUNT = 250
MAX_LINKED_SOURCE_BYTES = 50 * 1024 * 1024
MAX_LINKED_SOURCE_TOTAL_BYTES = 1024 * 1024 * 1024
LINKED_SOURCE_WORKERS = 4
OFFICIAL_HOSTS = {
    "bujhansi.ac.in",
    "www.bujhansi.ac.in",
    "academics.bujhansi.ac.in",
    "ums.bujhansi.ac.in",
}


@dataclass(frozen=True)
class UniversitySource:
    key: str
    title: str
    url: str
    kind: Literal["pdf", "webpage"]
    academic_year: str
    coverage: str

    @property
    def filename(self) -> str:
        extension = ".pdf" if self.kind == "pdf" else ".txt"
        return f"{self.key}{extension}"


UNIVERSITY_SOURCES = (
    UniversitySource(
        key="admission_bulletin_2026_27",
        title="BU Academic Bulletin and Updated Fees 2026-27",
        url=(
            "https://www.bujhansi.ac.in/media/s5yjfk2d/"
            "bulletin_2026-27-ii-update-fee-exam-fee-7.pdf"
        ),
        kind="pdf",
        academic_year="2026-27",
        coverage="programmes, admission, tuition fees and examination fees",
    ),
    UniversitySource(
        key="admission_guidelines_2026_27",
        title="BU Admission Information and Guidelines 2026",
        url=(
            "https://www.bujhansi.ac.in/media/kg2fd2d5/"
            "final-guidelines-2026-27.pdf"
        ),
        kind="pdf",
        academic_year="2026-27",
        coverage="admission process, dates and applicant instructions",
    ),
    UniversitySource(
        key="bca_syllabus",
        title="BU BCA Syllabus",
        url="https://www.bujhansi.ac.in/media/tmdezlty/bca.pdf",
        kind="pdf",
        academic_year="Not stated in filename; verify applicability",
        coverage="BCA syllabus and course structure",
    ),
    UniversitySource(
        key="btech_cse_syllabus",
        title="BU B.Tech CSE Syllabus",
        url=(
            "https://bujhansi.ac.in/media/3wophicd/"
            "b-tech-computer-science-and-engg.pdf"
        ),
        kind="pdf",
        academic_year=(
            "Scheme effective from 2017-18; current applicability "
            "is unverified"
        ),
        coverage="B.Tech Computer Science and Engineering syllabus",
    ),
    UniversitySource(
        key="mba_syllabus",
        title="BU MBA Syllabus",
        url="https://www.bujhansi.ac.in/media/t25llzvv/mba.pdf",
        kind="pdf",
        academic_year="Not stated in filename; verify applicability",
        coverage="MBA syllabus and course structure",
    ),
    UniversitySource(
        key="mca_syllabus",
        title="BU MCA Syllabus",
        url="https://www.bujhansi.ac.in/media/d0gnfcr1/mca.pdf",
        kind="pdf",
        academic_year="Not stated in filename; verify applicability",
        coverage="MCA syllabus and course structure",
    ),
    UniversitySource(
        key="admissions_page",
        title="BU Admissions and Current Admission Notices",
        url="https://www.bujhansi.ac.in/admissions/",
        kind="webpage",
        academic_year="Live page; verify each linked notice date",
        coverage="admission links and current admission notices",
    ),
    UniversitySource(
        key="syllabus_index",
        title="BU Official Syllabus Index",
        url="https://www.bujhansi.ac.in/academics/syllabus/",
        kind="webpage",
        academic_year="Live page; linked syllabus years may differ",
        coverage="official course and syllabus index",
    ),
    UniversitySource(
        key="examination_schemes",
        title="BU Examination Schemes",
        url="https://www.bujhansi.ac.in/academics/examination-schemes/",
        kind="webpage",
        academic_year="Session-specific; preserve the notice session",
        coverage="published examination schemes and schedules",
    ),
    UniversitySource(
        key="examination_notices",
        title="BU Examination Notices",
        url=(
            "https://www.bujhansi.ac.in/information-corner/"
            "examination-notices/"
        ),
        kind="webpage",
        academic_year="Live notices; verify each notice date",
        coverage="current examination announcements and links",
    ),
    UniversitySource(
        key="administration",
        title="BU Administration and University Authorities",
        url="https://www.bujhansi.ac.in/administration/",
        kind="webpage",
        academic_year="Live administration directory",
        coverage="university officers, authorities and administrative offices",
    ),
    UniversitySource(
        key="vice_chancellor",
        title="BU Vice-Chancellor",
        url="https://www.bujhansi.ac.in/administration/vice-chancellor/",
        kind="webpage",
        academic_year="Live directory; verify current office holder",
        coverage="Vice-Chancellor profile and official contact information",
    ),
    UniversitySource(
        key="registrar",
        title="BU Registrar",
        url="https://www.bujhansi.ac.in/administration/registrar/",
        kind="webpage",
        academic_year="Live directory; verify current office holder",
        coverage="Registrar profile and official contact information",
    ),
    UniversitySource(
        key="finance_officer",
        title="BU Finance Officer",
        url="https://www.bujhansi.ac.in/administration/finance-officer/",
        kind="webpage",
        academic_year="Live directory; verify current office holder",
        coverage="Finance Officer profile and official contact information",
    ),
    UniversitySource(
        key="controller_of_examinations",
        title="BU Controller of Examinations",
        url=(
            "https://www.bujhansi.ac.in/administration/"
            "controller-of-examinations/"
        ),
        kind="webpage",
        academic_year="Live directory; verify current office holder",
        coverage="Controller of Examinations profile and contacts",
    ),
    UniversitySource(
        key="dean_hod",
        title="BU Deans and Heads of Departments",
        url="https://www.bujhansi.ac.in/administration/dean-hod/",
        kind="webpage",
        academic_year="Live directory; verify current office holders",
        coverage="Deans, Directors and Heads of Departments",
    ),
    UniversitySource(
        key="campus_faculty",
        title="BU Campus Faculty and Department Heads",
        url=(
            "https://ums.bujhansi.ac.in/BUJhansi/"
            "frmViewcampusFaculty.aspx?cd=NAAxADkA"
        ),
        kind="webpage",
        academic_year="Live directory; verify current designation",
        coverage="campus faculty directory and published Heads of Departments",
    ),
    UniversitySource(
        key="training_placement",
        title="BU Training and Placement Cell",
        url="https://ums.bujhansi.ac.in/BUJhansi/Placement/index.html",
        kind="webpage",
        academic_year="Live cell information; individual records may be historical",
        coverage=(
            "placement-cell role, student training, internships, "
            "placement facilities and officials"
        ),
    ),
    UniversitySource(
        key="training_placement_procedure",
        title="BU Placement Procedure",
        url=(
            "https://ums.bujhansi.ac.in/BUJhansi/Placement/"
            "placement-procedure.html"
        ),
        kind="webpage",
        academic_year="Live procedure page; verify current drive instructions",
        coverage=(
            "company registration, job descriptions, student registration, "
            "shortlisting, interviews and offer process"
        ),
    ),
    UniversitySource(
        key="training_placement_for_students",
        title="BU Training and Placement Guidance for Students",
        url=(
            "https://ums.bujhansi.ac.in/BUJhansi/Placement/"
            "for-student.html"
        ),
        kind="webpage",
        academic_year="Live student guidance; verify current forms and policy",
        coverage=(
            "student login procedure, CV proforma and university "
            "training-and-placement policy"
        ),
    ),
    UniversitySource(
        key="training_placement_records",
        title="BU Published Placement Records",
        url=(
            "https://ums.bujhansi.ac.in/BUJhansi/Placement/"
            "frmPlacementRecord.aspx"
        ),
        kind="webpage",
        academic_year="Reporting period is not stated on the page",
        coverage="company-wise placed-candidate counts published by BU",
    ),
    UniversitySource(
        key="training_placement_drives",
        title="BU Placement Drive Notices",
        url=(
            "https://ums.bujhansi.ac.in/BUJhansi/Placement/"
            "frmPlacementDrive.aspx"
        ),
        kind="webpage",
        academic_year="Historical entries dated 2016-17; not current drives",
        coverage=(
            "historical company placement drives, registration dates "
            "and published openings"
        ),
    ),
    UniversitySource(
        key="chief_vigilance_officer",
        title="BU Chief Vigilance Officer",
        url="https://bujhansi.ac.in/administration/chief-vigilance-officer/",
        kind="webpage",
        academic_year="Live directory; verify current office holder",
        coverage="Chief Vigilance Officer and official contact information",
    ),
    UniversitySource(
        key="ombudsperson",
        title="BU Ombudsperson",
        url="https://bujhansi.ac.in/administration/ombudsperson/",
        kind="webpage",
        academic_year="Live directory; verify current office holder",
        coverage="Ombudsperson and student grievance information",
    ),
    UniversitySource(
        key="chancellor",
        title="BU Chancellor",
        url="https://www.bujhansi.ac.in/administration/chancellor/",
        kind="webpage",
        academic_year="Live directory; verify current office holder",
        coverage="Chancellor profile and official information",
    ),
    UniversitySource(
        key="dean_student_welfare",
        title="BU Dean of Students Welfare",
        url="https://bujhansi.ac.in/student-life/dean-student-welfare/",
        kind="webpage",
        academic_year="Live directory; verify current office holder",
        coverage="Dean of Students Welfare and student-support contacts",
    ),
    UniversitySource(
        key="examination_formats",
        title="BU Examination Formats",
        url="https://www.bujhansi.ac.in/academics/examination-formats/",
        kind="webpage",
        academic_year="Live page; verify the applicable session",
        coverage="examination forms and formats",
    ),
    UniversitySource(
        key="hostel_fees",
        title="BU Hostel Fee Information",
        url="https://www.bujhansi.ac.in/student-life/hostels-facilities/hostel-fee/",
        kind="webpage",
        academic_year="Live page; confirm fee session before quoting rates",
        coverage="hostel fee information",
    ),
    UniversitySource(
        key="hostel_charges",
        title="BU Hostel Charges",
        url="https://bujhansi.ac.in/media/ulafdkj1/638715099957836008.pdf",
        kind="pdf",
        academic_year="Memo dated 18 July 2019; verify current applicability",
        coverage="hostel charges and fee schedule; historical rate memo",
    ),
    UniversitySource(
        key="student_welfare",
        title="BU Dean of Students Welfare",
        url="https://bujhansi.ac.in/student-life/dean-student-welfare/",
        kind="webpage",
        academic_year="Live student-welfare information",
        coverage="student support and welfare information",
    ),
    UniversitySource(
        key="results",
        title="BU Student Results",
        url="https://www.bujhansi.ac.in/student-life/results/",
        kind="webpage",
        academic_year="Live student portal links",
        coverage="official result links and information",
    ),
    UniversitySource(
        key="central_library",
        title="BU Central Library",
        url="https://bujhansi.ac.in/academics/central-library/overview/",
        kind="webpage",
        academic_year="Live library information",
        coverage="central-library facilities and services",
    ),
    UniversitySource(
        key="library_ebooks",
        title="BU Central Library E-Content",
        url="https://bujhansi.ac.in/academics/central-library/e-books-library/",
        kind="webpage",
        academic_year="Live library links",
        coverage="official e-content and library links",
    ),
    UniversitySource(
        key="nss",
        title="BU National Service Scheme",
        url="https://www.bujhansi.ac.in/student-life/nss/",
        kind="webpage",
        academic_year="Live student-life information",
        coverage="National Service Scheme information",
    ),
    UniversitySource(
        key="ncc",
        title="BU National Cadet Corps",
        url="https://www.bujhansi.ac.in/student-life/ncc/",
        kind="webpage",
        academic_year="Live student-life information",
        coverage="National Cadet Corps information",
    ),
    UniversitySource(
        key="iqac",
        title="BU Internal Quality Assurance Cell",
        url="https://www.bujhansi.ac.in/academics/iqac/",
        kind="webpage",
        academic_year="Live university information",
        coverage="official quality-assurance information",
    ),
    UniversitySource(
        key="contact_directory",
        title="BU Contact Directory",
        url="https://www.bujhansi.ac.in/contact-directory/",
        kind="webpage",
        academic_year="Live directory; confirm details before relying on them",
        coverage="university office contacts",
    ),
    UniversitySource(
        key="department_directory",
        title="BU Institutes and Departments Directory",
        url="https://academics.bujhansi.ac.in/admin/view/instituteslist.html",
        kind="webpage",
        academic_year="Live directory",
        coverage="official institutes and department listing",
    ),
    UniversitySource(
        key="fee_refund_policy",
        title="BU Admission Fee Refund Policy",
        url=(
            "https://bujhansi.ac.in/media/vcjlzwx0/"
            "bu-jhansi_admission-fee-refund-policy.pdf"
        ),
        kind="pdf",
        academic_year="Policy PDF; confirm current applicability",
        coverage="admission fee refund terms and process",
    ),
    UniversitySource(
        key="ug_three_year_ordinance",
        title="BU Three-Year UG Programme Ordinance",
        url=(
            "https://bujhansi.ac.in/media/kvdjmc0i/"
            "ordinancefor3yearhonorsugprogrambu.pdf"
        ),
        kind="pdf",
        academic_year="Ordinance PDF; confirm current applicability",
        coverage="three-year undergraduate programme rules",
    ),
    UniversitySource(
        key="ug_four_year_ordinance",
        title="BU Four-Year UG Programme Ordinance",
        url=(
            "https://bujhansi.ac.in/media/itsh4xkg/"
            "ordinancefor4yearhonorsugprogrambu.pdf"
        ),
        kind="pdf",
        academic_year="Ordinance PDF; confirm current applicability",
        coverage="four-year undergraduate programme rules",
    ),
    UniversitySource(
        key="phd_admission_notice_2026_27",
        title="BU PhD Admission Notice 2026-27",
        url=(
            "https://bujhansi.ac.in/media/pi2jxxza/"
            "bu-jhansi-phd-admission-notice-2026-27.pdf"
        ),
        kind="pdf",
        academic_year="2026-27",
        coverage="PhD admission notice and dates",
    ),
    UniversitySource(
        key="phd_admission_guidelines_2026_27",
        title="BU PhD Admission Guidelines 2026-27",
        url=(
            "https://bujhansi.ac.in/media/lfbbmdsw/"
            "bu-jhansi-phd-guidelines-2026-27.pdf"
        ),
        kind="pdf",
        academic_year="2026-27",
        coverage="PhD admission process and applicant guidelines",
    ),
    UniversitySource(
        key="hostel_rules_2021",
        title="BU Hostel Rules and Guidelines",
        url="https://www.bujhansi.ac.in/pdf/hostel_rule_book_2021.pdf",
        kind="pdf",
        academic_year="2021 document; current applicability unverified",
        coverage="hostel conduct rules and student responsibilities",
    ),
    UniversitySource(
        key="student_disability_policy",
        title="BU Policy Related to Students with Disability",
        url=(
            "https://bujhansi.ac.in/media/m23j0iyr/"
            "2-1-2-policy-related-to-student-with-disability.pdf"
        ),
        kind="pdf",
        academic_year="Policy PDF; confirm current applicability",
        coverage="university policy related to students with disability",
    ),
    UniversitySource(
        key="fee_refund_page",
        title="BU Fee Refund Policy Page",
        url="https://www.bujhansi.ac.in/admissions/fee-refund-policy/",
        kind="webpage",
        academic_year="Live page; confirm the policy document's session",
        coverage="fee-refund policy information and official document links",
    ),
    UniversitySource(
        key="notice_section",
        title="BU General Notice Section",
        url="https://www.bujhansi.ac.in/information-corner/notice-section/",
        kind="webpage",
        academic_year="Live notices; verify each notice date",
        coverage="general university notices and official notice links",
    ),
    UniversitySource(
        key="student_clubs",
        title="BU Student Clubs Directory",
        url=(
            "https://www.bujhansi.ac.in/information-corner/"
            "student-clubs-directory/"
        ),
        kind="webpage",
        academic_year="Live student-life information",
        coverage="student clubs and extracurricular activities",
    ),
    UniversitySource(
        key="anti_ragging",
        title="BU Anti-Ragging Cell",
        url="https://www.bujhansi.ac.in/student-life/anti-ragging-cell/",
        kind="webpage",
        academic_year="Live student-support information",
        coverage="anti-ragging support and official contacts",
    ),
    UniversitySource(
        key="health_facilities",
        title="BU Health Facilities",
        url="https://www.bujhansi.ac.in/student-life/health-facilities/",
        kind="webpage",
        academic_year="Live student-life information",
        coverage="campus health facilities and services",
    ),
    UniversitySource(
        key="equal_opportunity",
        title="BU Equal Opportunity Cell",
        url=(
            "https://www.bujhansi.ac.in/student-life/"
            "equal-opportunity-cell/"
        ),
        kind="webpage",
        academic_year="Live student-support information",
        coverage="equal-opportunity support and related contacts",
    ),
    UniversitySource(
        key="sedg_cell",
        title="BU Socio-Economically Disadvantaged Groups Cell",
        url=(
            "https://www.bujhansi.ac.in/student-life/"
            "socio-economically-disadvantaged-groups-sedg-cell/"
        ),
        kind="webpage",
        academic_year="Live student-support information",
        coverage="support information for socio-economically disadvantaged groups",
    ),
    UniversitySource(
        key="hostel_facilities",
        title="BU Hostels and Facilities",
        url="https://www.bujhansi.ac.in/student-life/hostels-facilities/",
        kind="webpage",
        academic_year="Live student-life information",
        coverage="hostel facilities and official hostel information",
    ),
    UniversitySource(
        key="hostel_rules_page",
        title="BU Hostel Rules and Guidelines Page",
        url=(
            "https://www.bujhansi.ac.in/student-life/"
            "hostels-facilities/hostel-rules-and-guidelines/"
        ),
        kind="webpage",
        academic_year="Live page; linked rulebook is dated 2021",
        coverage="hostel rules page and official rulebook link",
    ),
    UniversitySource(
        key="campus_cyber_cell",
        title="BU Campus Cyber Cell",
        url="https://www.bujhansi.ac.in/student-life/campus-cyber-cell/",
        kind="webpage",
        academic_year="Live student-support information",
        coverage="campus cyber-safety support and contacts",
    ),
    UniversitySource(
        key="fyup_overview",
        title="BU Four-Year Undergraduate Programme Information",
        url="https://www.bujhansi.ac.in/academics/fyup/",
        kind="webpage",
        academic_year="Live academic information; verify current session",
        coverage="FYUP overview and official programme links",
    ),
    UniversitySource(
        key="fyup_campus",
        title="BU FYUP Professional Campus Courses",
        url="https://www.bujhansi.ac.in/academics/fyup/fyup-campus/",
        kind="webpage",
        academic_year="Live academic information; verify current session",
        coverage="FYUP professional courses offered on campus",
    ),
    UniversitySource(
        key="fyup_college",
        title="BU FYUP Traditional and College Courses",
        url="https://www.bujhansi.ac.in/academics/fyup/fyup-college/",
        kind="webpage",
        academic_year="Live academic information; verify current session",
        coverage="FYUP traditional courses and affiliated-college links",
    ),
    UniversitySource(
        key="affiliated_colleges",
        title="BU Affiliated Colleges",
        url="https://www.bujhansi.ac.in/about-bu/affiliated-colleges/",
        kind="webpage",
        academic_year="Live directory; verify current affiliation",
        coverage="official affiliated-college directory",
    ),
)


class _PageTextExtractor(HTMLParser):
    _ignored_tags = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._ignored_depth = 0
        self._anchor_url = ""
        self._anchor_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._ignored_tags:
            self._ignored_depth += 1
        elif tag == "a" and not self._ignored_depth:
            self._anchor_url = dict(attrs).get("href") or ""
            self._anchor_parts = []
        elif tag in {"p", "br", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "a" and self._anchor_url:
            text = " ".join(" ".join(self._anchor_parts).split())
            if text:
                self.links.append((text, self._anchor_url))
            self._anchor_url = ""
            self._anchor_parts = []
        elif tag in {"p", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            text = data.strip()
            if text:
                self.parts.append(text)
                if self._anchor_url:
                    self._anchor_parts.append(text)

    def to_text(self, base_url: str) -> str:
        body = " ".join(" ".join(self.parts).split())
        links = []
        for label, href in self.links:
            target = urljoin(base_url, href)
            host = urlparse(target).hostname or ""
            if host in OFFICIAL_HOSTS:
                links.append(f"- {label}: {target}")
        link_text = "\n".join(dict.fromkeys(links))
        if link_text:
            body = f"{body}\n\nOfficial links on this page:\n{link_text}"
        return body[:MAX_SOURCE_TEXT_CHARACTERS]


class _TableLinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, list[dict[str, list[str]]]]] = []
        self._row: dict[str, list[dict[str, list[str]]]] | None = None
        self._cell: dict[str, list[str]] | None = None
        self._anchor: dict[str, str | list[str]] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag == "tr":
            self._row = {"cells": []}
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = {"text": [], "links": []}
        elif tag == "a" and self._cell is not None:
            self._anchor = {
                "href": dict(attrs).get("href") or "",
                "text": [],
            }

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._anchor is not None:
            anchor_text = self._anchor["text"]
            if isinstance(anchor_text, list):
                anchor_text.append(text)
        elif self._cell is not None:
            self._cell["text"].append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor is not None and self._cell is not None:
            anchor_text = self._anchor["text"]
            href = self._anchor["href"]
            if isinstance(anchor_text, list) and isinstance(href, str):
                self._cell["links"].append(
                    {
                        "text": [" ".join(" ".join(anchor_text).split())],
                        "href": [href],
                    }
                )
            self._anchor = None
        elif tag in {"td", "th"} and self._cell is not None:
            if self._row is not None:
                self._row["cells"].append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _fetch_official_page(url: str) -> requests.Response:
    _assert_official_url(url)
    response = requests.get(url, timeout=(5, 20), allow_redirects=True)
    response.raise_for_status()
    _assert_official_url(response.url)
    if len(response.content) > MAX_SOURCE_BYTES:
        raise ValueError("The official source page exceeds 20 MB.")
    return response


def _linked_pdf_records(
    page_url: str,
    section: Literal["syllabus", "exam"],
) -> list[dict[str, str]]:
    response = _fetch_official_page(page_url)
    parser = _TableLinkExtractor()
    parser.feed(response.text)
    records = []
    known_urls = {
        source.url
        for source in UNIVERSITY_SOURCES
    }

    for row in parser.rows:
        cells = row["cells"]
        cell_text = [
            " ".join(cell["text"]).strip()
            for cell in cells
        ]
        row_text = [text for text in cell_text if text]
        for cell in cells:
            for link in cell["links"]:
                label = " ".join(link["text"]).strip()
                href = "".join(link["href"])
                target_url = urljoin(response.url, href)
                parsed = urlparse(target_url)
                if (
                    parsed.scheme != "https"
                    or parsed.hostname not in OFFICIAL_HOSTS
                    or not parsed.path.casefold().endswith(".pdf")
                    or target_url in known_urls
                ):
                    continue

                if section == "syllabus":
                    meaningful = [
                        value
                        for value in row_text
                        if not value.isdecimal()
                    ]
                    if len(meaningful) < 2:
                        continue
                    department, programme = meaningful[:2]
                    title = (
                        f"BU Syllabus — {programme} — {department}"
                    )
                    academic_year = (
                        "Live official syllabus index; session not stated "
                        "for this entry"
                    )
                    coverage = (
                        f"Course syllabus for {programme}, "
                        f"{department}"
                    )
                else:
                    title_parts = [
                        value
                        for value in row_text
                        if value.casefold() not in {
                            "download",
                            "click here",
                            "click here to view / download",
                        }
                    ]
                    title = (
                        "BU Examination Notice or Scheme — "
                        + " — ".join(title_parts)
                        if title_parts
                        else f"BU Examination Document — {label}"
                    )
                    year_match = re.search(
                        r"20\d{2}\s*[-–/]\s*\d{2,4}",
                        title,
                    )
                    academic_year = (
                        year_match.group(0)
                        if year_match
                        else "Linked notice; verify its published date"
                    )
                    coverage = (
                        "Official examination schedule or notice; "
                        "check for a newer correction"
                    )

                records.append(
                    {
                        "title": title,
                        "url": target_url,
                        "academic_year": academic_year,
                        "coverage": coverage,
                        "kind": section,
                    }
                )
                known_urls.add(target_url)
                if len(records) >= MAX_LINKED_SOURCE_COUNT:
                    return records
    return records


def load_university_documents(
    data_dir: Path = UNIVERSITY_DATA_DIR,
    modified_after: float | None = None,
    force_source_keys: set[str] | None = None,
) -> list[Document]:
    documents: list[Document] = []
    force_source_keys = force_source_keys or set()
    for source in UNIVERSITY_SOURCES:
        source_path = data_dir / source.filename
        if not source_path.is_file():
            continue
        metadata_path = source_path.with_suffix(
            source_path.suffix + ".json"
        )
        if (
            modified_after is not None
            and source.key not in force_source_keys
            and max(
                source_path.stat().st_mtime,
                (
                    metadata_path.stat().st_mtime
                    if metadata_path.is_file()
                    else 0
                ),
            )
            <= modified_after
        ):
            continue

        metadata = {
            "source": source.title,
            "source_url": source.url,
            "academic_year": source.academic_year,
            "coverage": source.coverage,
            "source_key": source.key,
            "source_kind": source.kind,
        }
        if source.kind == "pdf":
            loaded_pages = PyPDFLoader(str(source_path)).load()
            if source.key == "btech_cse_syllabus":
                with pymupdf.open(source_path) as pdf:
                    for page in loaded_pages:
                        if page.page_content.strip():
                            continue
                        page_number = page.metadata.get("page")
                        if not isinstance(page_number, int):
                            continue
                        pdf_page = pdf[page_number]
                        text_page = pdf_page.get_textpage_ocr(
                            language="eng",
                            dpi=200,
                            full=True,
                        )
                        page.page_content = pdf_page.get_text(
                            textpage=text_page
                        )
            text_pages = [
                Document(
                    page_content=page.page_content,
                    metadata={**page.metadata, **metadata},
                )
                for page in loaded_pages
                if page.page_content.strip()
            ]
            documents.extend(text_pages)
            if not text_pages:
                ocr_path = data_dir / f"{source.key}.ocr.txt"
                ocr_metadata_path = ocr_path.with_suffix(
                    ocr_path.suffix + ".json"
                )
                pdf_metadata_path = source_path.with_suffix(
                    source_path.suffix + ".json"
                )
                if (
                    ocr_path.is_file()
                    and ocr_metadata_path.is_file()
                    and pdf_metadata_path.is_file()
                ):
                    ocr_metadata = json.loads(
                        ocr_metadata_path.read_text(encoding="utf-8")
                    )
                    pdf_metadata = json.loads(
                        pdf_metadata_path.read_text(encoding="utf-8")
                    )
                    if (
                        ocr_metadata.get("source_sha256")
                        == pdf_metadata.get("sha256")
                    ):
                        ocr_text = ocr_path.read_text(
                            encoding="utf-8"
                        ).strip()
                        if ocr_text:
                            documents.append(
                                Document(
                                    page_content=ocr_text,
                                    metadata={**metadata, "page": 0},
                                )
                            )
        else:
            text = source_path.read_text(encoding="utf-8").strip()
            if text:
                documents.append(
                    Document(page_content=text, metadata=metadata)
                )

    catalog_urls = {
        source.url
        for source in UNIVERSITY_SOURCES
        if source.kind == "pdf"
        and (data_dir / source.filename).is_file()
    }
    for section in ("syllabi", "exam_schedules"):
        section_dir = data_dir / section
        if not section_dir.is_dir():
            continue
        for pdf_path in sorted(section_dir.glob("*.pdf")):
            metadata_path = pdf_path.with_suffix(".json")
            if not metadata_path.is_file():
                continue
            if (
                modified_after is not None
                and max(
                    pdf_path.stat().st_mtime,
                    metadata_path.stat().st_mtime,
                )
                <= modified_after
            ):
                continue
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            source_url = metadata.get("url", "")
            _assert_official_url(source_url)
            if source_url in catalog_urls:
                continue
            if (
                metadata.get("sha256")
                != hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            ):
                continue
            loaded_pages = PyPDFLoader(str(pdf_path)).load()
            documents.extend(
                Document(
                    page_content=page.page_content,
                    metadata={
                        **page.metadata,
                        "source": metadata["title"],
                        "source_url": source_url,
                        "academic_year": metadata["academic_year"],
                        "coverage": metadata["coverage"],
                        "source_key": f"{section}:{pdf_path.stem}",
                        "source_kind": "pdf",
                    },
                )
                for page in loaded_pages
                if page.page_content.strip()
            )

    if not documents:
        raise FileNotFoundError(
            "No Bundelkhand University source text is installed. Run "
            "'python -m backend.refresh_university_sources' to download "
            "the official source pack."
        )
    return documents


def get_university_source_status(
    data_dir: Path = UNIVERSITY_DATA_DIR,
) -> list[dict[str, str | bool | None]]:
    records = []
    for source in UNIVERSITY_SOURCES:
        source_path = data_dir / source.filename
        metadata_path = source_path.with_suffix(source_path.suffix + ".json")
        metadata: dict[str, str] = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records.append(
            {
                "title": source.title,
                "url": source.url,
                "kind": source.kind,
                "academic_year": source.academic_year,
                "coverage": source.coverage,
                "available": source_path.is_file()
                and source_path.stat().st_size > 0,
                "last_fetched": metadata.get("fetched_at"),
                "last_modified": metadata.get("last_modified"),
            }
        )
    for section in ("syllabi", "exam_schedules"):
        section_dir = data_dir / section
        if not section_dir.is_dir():
            continue
        for metadata_path in sorted(section_dir.glob("*.json")):
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            pdf_path = metadata_path.with_suffix(".pdf")
            source_url = metadata.get("url", "")
            try:
                _assert_official_url(source_url)
            except ValueError:
                available = False
            else:
                available = (
                    pdf_path.is_file()
                    and pdf_path.stat().st_size > 0
                    and len(metadata.get("sha256", "")) == 64
                )
            records.append(
                {
                    "title": metadata["title"],
                    "url": source_url,
                    "kind": "pdf",
                    "academic_year": metadata["academic_year"],
                    "coverage": metadata["coverage"],
                    "available": available,
                    "last_fetched": metadata.get("fetched_at"),
                    "last_modified": metadata.get("last_modified"),
                }
            )
    return records


def _assert_official_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS:
        raise ValueError(
            f"Refusing to fetch a non-official university URL: {url}"
        )


def refresh_university_sources(
    data_dir: Path = UNIVERSITY_DATA_DIR,
    source_keys: set[str] | None = None,
) -> list[dict[str, str]]:
    sources = UNIVERSITY_SOURCES
    if source_keys is not None:
        known_keys = {source.key for source in UNIVERSITY_SOURCES}
        unknown_keys = source_keys - known_keys
        if unknown_keys:
            raise ValueError(
                "Unknown university source key(s): "
                + ", ".join(sorted(unknown_keys))
            )
        sources = tuple(
            source
            for source in UNIVERSITY_SOURCES
            if source.key in source_keys
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []

    for source in sources:
        _assert_official_url(source.url)
        destination = data_dir / source.filename
        try:
            response = requests.get(
                source.url,
                timeout=(5, 20),
                allow_redirects=True,
            )
            response.raise_for_status()
            final_url = response.url
            _assert_official_url(final_url)
            content = response.content
            if not content or len(content) > MAX_SOURCE_BYTES:
                raise ValueError("Source response is empty or exceeds 20 MB.")

            if source.kind == "pdf":
                if not content.startswith(b"%PDF-"):
                    raise ValueError(
                        "The university source did not return a PDF file."
                    )
                output = content
            else:
                extractor = _PageTextExtractor()
                extractor.feed(response.text)
                output = extractor.to_text(source.url).encode("utf-8")
                if len(output.strip()) < 80:
                    raise ValueError(
                        "Could not extract enough text from the university page."
                    )

            temporary_path = destination.with_suffix(
                destination.suffix + ".tmp"
            )
            temporary_path.write_bytes(output)
            temporary_path.replace(destination)
            metadata = {
                "title": source.title,
                "url": source.url,
                "final_url": final_url,
                "academic_year": source.academic_year,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "last_modified": response.headers.get("Last-Modified", ""),
                "sha256": hashlib.sha256(output).hexdigest(),
            }
            destination.with_suffix(
                destination.suffix + ".json"
            ).write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            results.append({"title": source.title, "status": "updated"})
        except (requests.RequestException, OSError, ValueError) as error:
            results.append(
                {"title": source.title, "status": f"error: {error}"}
            )

    return results


def refresh_linked_university_documents(
    data_dir: Path = UNIVERSITY_DATA_DIR,
) -> list[dict[str, str]]:
    sections = (
        (
            "syllabi",
            "https://www.bujhansi.ac.in/academics/syllabus/",
        ),
        (
            "exam_schedules",
            "https://www.bujhansi.ac.in/academics/examination-schemes/",
        ),
        (
            "exam_schedules",
            "https://www.bujhansi.ac.in/information-corner/"
            "examination-notices/",
        ),
    )
    records_by_url: dict[str, dict[str, str]] = {}
    results: list[dict[str, str]] = []
    for section, page_url in sections:
        try:
            records = _linked_pdf_records(
                page_url,
                "syllabus" if section == "syllabi" else "exam",
            )
        except (requests.RequestException, ValueError) as error:
            results.append(
                {
                    "title": page_url,
                    "status": f"error: {error}",
                }
            )
            continue
        for record in records:
            records_by_url.setdefault(record["url"], record)

    stored_bytes = sum(
        pdf_path.stat().st_size
        for section in ("syllabi", "exam_schedules")
        for pdf_path in (data_dir / section).glob("*.pdf")
        if (data_dir / section).is_dir()
    )
    size_lock = Lock()
    records_to_download = []
    for record in records_by_url.values():
        section_name = (
            "syllabi"
            if record["kind"] == "syllabus"
            else "exam_schedules"
        )
        section_dir = data_dir / section_name
        filename = hashlib.sha256(
            record["url"].encode("utf-8")
        ).hexdigest()[:20]
        destination = section_dir / f"{filename}.pdf"
        metadata_path = destination.with_suffix(".json")
        if destination.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("url") == record["url"]
                and metadata.get("sha256")
                == hashlib.sha256(destination.read_bytes()).hexdigest()
            ):
                results.append(
                    {"title": record["title"], "status": "current"}
                )
                continue
        if stored_bytes >= MAX_LINKED_SOURCE_TOTAL_BYTES:
            results.append(
                {
                    "title": record["title"],
                    "status": (
                        "error: linked document cache reached the "
                        "500 MB safety limit"
                    ),
                }
            )
            continue
        records_to_download.append((record, destination))

    def download_record(
        item: tuple[dict[str, str], Path],
    ) -> dict[str, str]:
        nonlocal stored_bytes
        record, destination = item
        temporary_path: Path | None = None
        response: requests.Response | None = None
        storage_adjustment = 0
        try:
            response = requests.get(
                record["url"],
                timeout=(5, 20),
                allow_redirects=True,
                stream=True,
            )
            response.raise_for_status()
            _assert_official_url(response.url)
            content_length = response.headers.get("Content-Length")
            if (
                content_length
                and int(content_length) > MAX_LINKED_SOURCE_BYTES
            ):
                raise ValueError(
                    "Linked source exceeds the 50 MB per-file safety limit."
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            prefix = bytearray()
            downloaded_bytes = 0
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f"{destination.stem}.",
                suffix=".pdf.tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > MAX_LINKED_SOURCE_BYTES:
                        raise ValueError(
                            "Linked source exceeds the 50 MB per-file "
                            "safety limit."
                        )
                    if len(prefix) < 5:
                        prefix.extend(chunk[: 5 - len(prefix)])
                    digest.update(chunk)
                    temporary_file.write(chunk)

            if not bytes(prefix).startswith(b"%PDF-"):
                raise ValueError("Linked source is not a valid PDF.")
            if content_length and downloaded_bytes != int(content_length):
                raise ValueError("Linked PDF download was incomplete.")

            previous_size = (
                destination.stat().st_size
                if destination.is_file()
                else 0
            )
            with size_lock:
                if (
                    stored_bytes - previous_size + downloaded_bytes
                    > MAX_LINKED_SOURCE_TOTAL_BYTES
                ):
                    raise ValueError(
                        "Linked document cache reached the 1 GB safety limit."
                    )
                storage_adjustment = downloaded_bytes - previous_size
                stored_bytes += storage_adjustment

            temporary_path.replace(destination)
            temporary_path = None
            metadata = {
                **record,
                "final_url": response.url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "last_modified": response.headers.get(
                    "Last-Modified",
                    "",
                ),
                "sha256": digest.hexdigest(),
            }
            destination.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            return {"title": record["title"], "status": "updated"}
        except (requests.RequestException, OSError, ValueError) as error:
            if storage_adjustment:
                with size_lock:
                    stored_bytes -= storage_adjustment
            return {
                "title": record["title"],
                "status": f"error: {error}",
            }
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if response is not None:
                response.close()

    with ThreadPoolExecutor(
        max_workers=LINKED_SOURCE_WORKERS
    ) as executor:
        futures = [
            executor.submit(download_record, item)
            for item in records_to_download
        ]
        results.extend(future.result() for future in as_completed(futures))
    return results
