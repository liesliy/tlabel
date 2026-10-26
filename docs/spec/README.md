# TLabel Specification Directory

This directory holds the **normative** specification for the TLabel data format.

## The normative document

- **[`tlabel-spec-v1.md`](./tlabel-spec-v1.md)** — **TLabel Data Format
  Specification v1 (Schema V2, 14 dimensions).** This is the single normative
  source of truth ("正本"). It is written in ROS REP style: numbered sections,
  numbered normative clauses (e.g. `S-4.3-1`), and RFC 2119 keywords
  (`MUST` / `SHOULD` / `MAY`) in uppercase. It defines the 14 Schema V2
  dimensions, the L1–L4 Compliance Level rules, adapter conformance,
  serialization, versioning, and the extension mechanism.

Where any other document conflicts with `tlabel-spec-v1.md` (or with the code
it is derived from), **the specification and the code govern**. See §11 of the
specification for the full conflict list.

## How the other documents relate

| Document | Kind | Role |
|---|---|---|
| `docs/spec/tlabel-spec-v1.md` | **Normative** | The specification (正本). Authoritative for all conformance questions. |
| `docs/tlabel-format.md` | Informative / tutorial | A field-by-field walkthrough of the format. Predates the spec; contains some stale enums and field names (see spec §11, D-04…D-13). Kept for readability, not for conformance. |
| `docs/annotation-spec.md` | Informative / tutorial | Annotation guidelines and validation-rule narrative. Some rules are aspirational and not enforced by the reference validator (spec §11, D-09…D-12). |
| `docs/DESIGN.md` | Informative | Early design notes. Contains stale "12-dimension" captions (spec §11, D-14, D-15). |
| `docs/TLabel_Design_Document.md` | Informative (design rationale) | The design rationale and paper-narrative guide. Useful for "why", but not for "what is required". |
| `docs/whitepaper.md` | Informative (obsolete) | Pre-v2.0 whitepaper describing the removed flat feature space. Superseded by the specification (spec §11, D-01). |
| `schema/tlabel-schema.json` | Machine artifact | JSON-level document schema. Partially out of sync with the adapters (spec §11, D-16). |
| `internal-docs/RFC-README.md` | Process | The RFC process that governs changes to the core schema. |

## Rule of thumb

- Need to know **what is required** → read `tlabel-spec-v1.md`.
- Need to know **how to use** the toolkit → read the tutorials.
- Need to know **why** it was designed this way → read
  `TLabel_Design_Document.md`.
- Need to change the format → follow `internal-docs/RFC-README.md`.
