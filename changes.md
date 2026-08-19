# Changes

## v1.0.4

### ILDG I/O

- Read and write both ILDG 32-bit and 64-bit floating-point payloads correctly,
  using the precision declared in the `ildg-format` metadata.
- Add `precision=:field`, `precision=32`, and `precision=64` to
  `save_binarydata`; `:field` preserves the gauge field's component precision.
- Emit ILDG v1.2 XML metadata with the required namespace and record ordering,
  and store the binary payload in big-endian byte order.
- Update the CLIME_jll command invocation and ensure that opened files and
  temporary resources are closed reliably.

### MPI and GPU correctness

- Restrict LIME extraction and packing to the MPI root rank, use the gauge
  field's communicator instead of assuming `MPI.COMM_WORLD`, and share unique
  temporary payload paths safely across ranks.
- Synchronize direct JACC/GPU writes, mark lattice data as modified, and refresh
  halo regions after loading a configuration.

### Tests

- Add 32-bit and 64-bit ILDG round-trip and metadata tests on CPU, MPI domain
  decompositions, and GPU/JACC backends.
- Add regression coverage for portable element types, halo epochs, and
  long-distance shifted-field operations.
