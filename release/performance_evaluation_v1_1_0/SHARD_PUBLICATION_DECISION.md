# Continuation-shard publication decision

**Decision: publish both** the full canonical continuation shards *and* the compact extracted SBS reference rows.

* The 289 shard artifacts (96 shard CSVs, their JSON manifests and correctness CSVs, and `RUN_SUMMARY.json`)
  total under 3 MB, are pure derived simulator outputs (latency statistics, state identifiers and request-id hashes;
  no trace content, prompts, credentials or local paths), and every file matches the frozen manifest
  `FRESH_LATENCY_ARTIFACT_HASHES_V1.json`. Neither size, privacy nor licensing argues against publishing them.
* They are the only artifact holding the `SBS_REFERENCE` rows, so they are the sole source from which the corrected
  derivative can be regenerated and the byte-identity proof of the correction replayed. The compact
  `SBS_REFERENCE_ROWS_V1.csv` alone lets a reader check the corrected columns but not the replay.
* The partial directory `fresh-latency-confirmatory-v1` of the local provenance archive (3 files, from an
  interrupted earlier attempt) is **not** canonical and is not released; only `fresh-latency-execution-v1` is.
