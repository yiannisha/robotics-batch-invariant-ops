# Recovered historical pilot inputs

Six distinct prepared/noise requests cover all 24 calls in the retained pilot. Each tensor is matched independently by dtype, shape and byte SHA-256 against the archived pilot, whose uncompressed JSON bytes match the historical exposure receipt. Replay uses retained actions and official preprocessing; no neural inference is used.

These files recover only this pilot category. The 1,172 historical source-corpus request records and remaining exposure categories are not established by this result. There is no final freshness seal or new batch-stability measurement. Raw model-free replay frames remain in the recovery directories on scratch; original pilot evidence remains unchanged.
