from rdflib.graph import Graph, Namespace

from data_science_agent.dtos.base import MetadataBase
from data_science_agent.graph import AgentState
from data_science_agent.language import import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE

Metadata = import_language_dto(AGENT_LANGUAGE, base_dto_class=MetadataBase)

RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDF_SCHEMA = Namespace("http://www.w3.org/2000/01/rdf-schema#")
DC = Namespace("http://purl.org/dc/terms/")
DCAT = Namespace("http://www.w3.org/ns/dcat#")
DCATDE = Namespace("http://dcat-ap.de/def/dcatde/")
ADMS = Namespace("http://www.w3.org/ns/adms#")
SPDX = Namespace("http://spdx.org/rdf/terms#")
VCARD = Namespace("http://www.w3.org/2006/vcard/ns#")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
LOCN = Namespace("http://www.w3.org/ns/locn#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
DATAEU = Namespace("http://data.europa.eu/r5r/")

excluded_predicates = {
    # --- Generic / structural
    RDF.type,
    RDF_SCHEMA.label,

    # --- Dublin Core: administrative or technically narrow
    DC.identifier,
    DC.format,
    DC.language,
    DC.license,
    DC.type,
    DC.issued,
    DC.modified,
    DC.conformsTo,
    DC.provenance,
    DC.relation,
    DC.relations,
    DC.hasVersion,
    DC.isVersionOf,
    DC.isReferencedBy,
    DC.references,
    DC.source,
    DC.rights,
    DC.creator,
    DC.spatial,

    # --- DCAT: technical distribution and access metadata
    DCAT.distribution,
    DCAT.downloadURL,
    DCAT.accessURL,
    DCAT.accessService,
    DCAT.servesDataset,
    DCAT.endpointURL,
    DCAT.endpointDescription,
    DCAT.mediaType,
    DCAT.byteSize,
    DCAT.compressFormat,
    DCAT.temporalResolution,
    DCAT.granularity,
    DCAT.theme,
    DCAT.contactPoint,

    # --- dcat‑ap.de extensions: legal and administrative detail
    DCATDE.licenseAttributionByText,
    DCATDE.contributorID,
    DCATDE.legalBasis,
    DCATDE.maintainer,
    DCATDE.originator,
    DCATDE.geocodingDescription,
    DCATDE.politicalGeocodingURI,
    DCATDE.politicalGeocodingLevelURI,
    DCATDE.plannedAvailability,
    DCATDE.qualityProcessURI,

    # --- FOAF / vCard contact and homepages
    FOAF.page,
    FOAF.homepage,
    FOAF.mbox,
    FOAF.name,
    VCARD.fn,
    VCARD.hasEmail,
    VCARD.hasTelephone,
    VCARD.hasURL,
    VCARD.hasCountryName,
    VCARD.hasLocality,
    VCARD.hasPostalCode,
    VCARD.hasStreetAddress,

    # --- ADMS / SPDX: identifiers, checksums, version notes
    ADMS.identifier,
    ADMS.status,
    ADMS.versionNotes,
    SPDX.checksum,

    # --- OWL / SKOS / LOCN: descriptive but not textual insight
    OWL.versionInfo,
    SKOS.prefLabel,
    LOCN.geometry,

    # --- EU vocab additions or categories that may repeat
    DATAEU.hvdCategory,
    DATAEU.availability,
    DATAEU.applicableLegislation,
}

excluded_predicates_uris = {str(p) for p in excluded_predicates}

# Manually remove DC.format
excluded_predicates_uris.add("http://purl.org/dc/terms/format")


@track_duration
def analyse_metadata(state: AgentState) -> AgentState:
    """
    Parse and analyse RDF metadata, excluding uninformative/technical predicates.
    Stores the remaining triples (as Metadata objects) in state['metadata'].
    """
    g = Graph()
    g.parse(state["metadata_path"], format="xml")

    metadata = []
    for s, p, o in g.triples((None, None, None)):
        if str(p) not in excluded_predicates_uris:
            metadata.append(Metadata(subject=str(s), predicate=str(p), object=str(o)))

    state["metadata"] = metadata

    print("Metadata (filtered):")
    for x in metadata:
        print(f"  {x.subject} -- {x.predicate} --> {x.object}")

    return state
