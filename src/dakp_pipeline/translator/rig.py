"""Reference Ingest Guide (RIG) generation (Milestone 8).

Produces DAKP RIG content matching the structure and conventions of
``../DINGO/src/translator_ingest/ingests/dakp/dakp_rig.yaml``: ``source_info``
(``infores:multiomics-drugapprovals``), ``supporting_data_source_info`` (DailyMed / FAERS /
MEDI), and ``target_info`` with ``edge_type_info`` (the three DAKP edge families) and
``node_type_info``.

The edge/node category contract is imported from :mod:`dakp_pipeline.translator.contract`
so the RIG and the KGX validator never drift apart. YAML is emitted with a small
dependency-free deterministic serializer (no PyYAML) — see :func:`rig_yaml`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dakp_pipeline.translator.contract import (
    CHEMICAL_DRUG_CATEGORIES,
    DISEASE_PHENOTYPE_CATEGORIES,
    EDGE_FAMILIES,
    INFORES_DAILYMED,
    INFORES_DAKP,
    INFORES_FAERS,
    INFORES_MEDI,
    PREDICATE_APPLIED_TO_TREAT,
    PREDICATE_CONTRAINDICATED_IN,
    PREDICATE_TREATS,
)

RIG_NAME = "Multiomics Drug Approvals Knowledge Provider (DAKP) Reference Ingest Guide"
TARGET_INFORES = "infores:translator-dakp-kgx"

_UI_EXPLANATIONS: dict[str, str] = {
    PREDICATE_TREATS: (
        "The 'treats' predicate indicates an FDA-approved therapeutic relationship between a drug "
        "and a disease, with supporting regulatory approval information."
    ),
    PREDICATE_APPLIED_TO_TREAT: (
        "The 'applied_to_treat' predicate indicates an therapeutic relationship between a drug and a "
        "disease, without FDA approval, based on usage reports in FAERS."
    ),
    PREDICATE_CONTRAINDICATED_IN: (
        "The 'contraindicated_in' predicate indicates a contraindication relationship between a drug "
        "and a disease or condition, based on warnings present in FDA approval documents."
    ),
}

_NODE_TYPE_INFO: tuple[dict[str, Any], ...] = (
    {"node_category": "biolink:ChemicalEntity", "source_identifier_types": ["CHEBI", "UNII"]},
    {"node_category": "biolink:SmallMolecule", "source_identifier_types": ["CHEBI", "UNII"]},
    {"node_category": "biolink:MolecularMixture", "source_identifier_types": ["CHEBI"]},
    {"node_category": "biolink:ComplexMolecularMixture", "source_identifier_types": ["CHEBI"]},
    {"node_category": "biolink:Disease", "source_identifier_types": ["MONDO"]},
    {"node_category": "biolink:PhenotypicFeature", "source_identifier_types": ["HP"]},
)


def _supporting_data_source_info() -> list[dict[str, Any]]:
    return [
        {
            "infores_id": INFORES_DAILYMED,
            "name": "DailyMed",
            "description": "DailyMed provides trustworthy information about marketed drugs in the United States",
            "terms_of_use_info": {"terms_of_use_description": "DailyMed data are freely available. The data are in the public domain."},
            "relevant_files": [
                {
                    "file_name": "DailyMed Data",
                    "location": "https://dailymed.nlm.nih.gov/dailymed/",
                    "description": "Structured product labeling documents",
                }
            ],
        },
        {
            "infores_id": INFORES_FAERS,
            "name": "FDA Adverse Event Reporting System (FAERS)",
            "description": "FAERS contains adverse event reports, medication error reports and product quality complaints",
            "terms_of_use_info": {"terms_of_use_description": "FAERS data files are in the public domain and available for download."},
            "relevant_files": [
                {
                    "file_name": "FAERS Quarterly Data Files",
                    "location": (
                        "https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/"
                        "fda-adverse-event-reporting-system-faers-quarterly-data-extract-files"
                    ),
                    "description": "Quarterly data files containing adverse event reports",
                }
            ],
        },
        {
            "infores_id": INFORES_MEDI,
            "name": "MEDI",
            "description": "MEDI contains contraindication information extracted from DailyMed documents",
            "terms_of_use_info": {"terms_of_use_description": "NA"},
            "relevant_files": [
                {
                    "file_name": "contraindicationList.xlsx",
                    "location": "https://github.com/everycure-org/matrix-indication-list/releases/",
                    "description": "Contraindications extracted from DailyMed",
                }
            ],
        },
    ]


def _source_info() -> dict[str, Any]:
    return {
        "infores_id": INFORES_DAKP,
        "name": "Multiomics Drug Approvals Knowledge Provider (Multiomics-DAKP)",
        "description": (
            "The Drug Approvals KP provides information on FDA-approved drugs and their approved indications, "
            "derived from DailyMed and FAERS data. Information includes drug-disease treatment relationships "
            "with FDA approval status, approval numbers, and supporting evidence from adverse event reports."
        ),
        "citations": [
            "Generating Biomedical Knowledge Graphs from Knowledge Bases, Registries, and Multiomic Data "
            "(preprint): (https://pmc.ncbi.nlm.nih.gov/articles/PMC11601480/"
        ],
        "terms_of_use_info": {"terms_of_use_description": "Freely available"},
        "data_access_locations": [
            "Direct download links for the latest version are given in the manifest file: "
            "https://github.com/multiomicsKP/drug_approvals_kp/blob/main/manifest.json"
        ],
        "data_provision_mechanisms": ["file_download"],
        "data_formats": ["kgx"],
        "data_versioning_and_releases": "New versions are released periodically",
    }


def _ingest_info() -> dict[str, Any]:
    return {
        "ingest_categories": ["translator_knowledge_creator"],
        "utility": (
            "DAKP provides curated drug-disease treatment relationships with FDA approval status and supporting "
            "evidence. This knowledge is essential for understanding approved therapeutic uses and clinical "
            "evidence. DAKP also provides information on drug use in the absence of FDA approval, and information "
            "on contraindications."
        ),
        "scope": (
            "FDA-approved drugs and their treatment relationships with diseases, including approval status, FDA "
            "approval numbers, and case counts from adverse event reporting; information on drug use in the absence "
            "of FDA approval, with case counts from adverse event reporting; and information on contraindications."
        ),
        "relevant_files": [
            {
                "file_name": "DAKP processed data",
                "location": "https://db.systemsbiology.net/gestalt/KG/",
                "description": "Drug approval data processed from DailyMed and FAERS",
            }
        ],
        "included_content": [
            {
                "file_name": "DAKP processed data",
                "included_records": "Drug-disease treatment relationships with FDA approval information",
                "fields_used": "drug identifiers, disease identifiers, FDA approval numbers, approval status, case counts, evidence IDs",
            }
        ],
    }


def _edge_type_info() -> list[dict[str, Any]]:
    """One entry per DAKP edge family, in canonical family order (contract.EDGE_FAMILIES)."""
    return [
        {
            "subject_categories": list(CHEMICAL_DRUG_CATEGORIES),
            "predicates": [family.predicate],
            "object_categories": list(DISEASE_PHENOTYPE_CATEGORIES),
            "knowledge_level": ["knowledge_assertion"],
            "agent_type": ["text_mining_agent"],
            "ui_explanation": _UI_EXPLANATIONS[family.predicate],
        }
        for family in EDGE_FAMILIES.values()
    ]


def _target_info() -> dict[str, Any]:
    return {"infores_id": TARGET_INFORES, "edge_type_info": _edge_type_info(), "node_type_info": [dict(entry) for entry in _NODE_TYPE_INFO]}


def generate_rig() -> dict[str, Any]:
    """Build the DAKP RIG content as an ordered mapping (DINGO ``dakp_rig.yaml`` structure)."""
    return {
        "name": RIG_NAME,
        "supporting_data_source_info": _supporting_data_source_info(),
        "source_info": _source_info(),
        "ingest_info": _ingest_info(),
        "target_info": _target_info(),
        "provenance_info": {
            "contributions": [
                "Gwenlyn Glusman - code author, domain expertise, data modeling",
                "Matthew Brush - data modeling",
                "Sierra Moxon - code, data modeling",
            ]
        },
    }


# --- dependency-free deterministic YAML serialization ----------------------------


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):  # bool before int: bool is an int subclass
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _is_nonempty_container(value: Any) -> bool:
    return isinstance(value, (dict, list)) and bool(value)


def _empty_inline(value: Any) -> str:
    return "{}" if isinstance(value, dict) else "[]"


def _emit_mapping(mapping: dict[str, Any], indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    for key, value in mapping.items():
        if _is_nonempty_container(value):
            lines.append(f"{pad}{key}:")
            _emit_node(value, indent + 1, lines)
        elif isinstance(value, (dict, list)):
            lines.append(f"{pad}{key}: {_empty_inline(value)}")
        else:
            lines.append(f"{pad}{key}: {_format_scalar(value)}")


def _emit_sequence(sequence: list[Any], indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    for item in sequence:
        if isinstance(item, dict) and item:
            entries = list(item.items())
            first_key, first_value = entries[0]
            if _is_nonempty_container(first_value):
                lines.append(f"{pad}- {first_key}:")
                _emit_node(first_value, indent + 2, lines)
            elif isinstance(first_value, (dict, list)):
                lines.append(f"{pad}- {first_key}: {_empty_inline(first_value)}")
            else:
                lines.append(f"{pad}- {first_key}: {_format_scalar(first_value)}")
            # Remaining keys align under the first (two spaces past the dash).
            for key, value in entries[1:]:
                if _is_nonempty_container(value):
                    lines.append(f"{pad}  {key}:")
                    _emit_node(value, indent + 2, lines)
                elif isinstance(value, (dict, list)):
                    lines.append(f"{pad}  {key}: {_empty_inline(value)}")
                else:
                    lines.append(f"{pad}  {key}: {_format_scalar(value)}")
        else:
            lines.append(f"{pad}- {_format_scalar(item)}")


def _emit_node(data: Any, indent: int, lines: list[str]) -> None:
    if isinstance(data, dict):
        _emit_mapping(data, indent, lines)
    else:
        _emit_sequence(data, indent, lines)


def rig_yaml(rig: Mapping[str, Any] | None = None) -> str:
    """Serialize a RIG mapping to deterministic block-style YAML (defaults to :func:`generate_rig`)."""
    lines: list[str] = []
    _emit_mapping(dict(rig if rig is not None else generate_rig()), 0, lines)
    return "\n".join(lines) + "\n"


def write_rig(path: Path, rig: Mapping[str, Any] | None = None) -> Path:
    """Write the RIG YAML to ``path`` (creating parent dirs) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rig_yaml(rig), encoding="utf-8")
    return path


def rig_text() -> str:
    """Return the generated DAKP RIG as a YAML string."""
    return rig_yaml()


__all__ = ["RIG_NAME", "TARGET_INFORES", "generate_rig", "rig_text", "rig_yaml", "write_rig"]
