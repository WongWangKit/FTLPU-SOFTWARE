# StableHLO reference interpreter compatibility

The optional HF frontend static-mask normalization uses
`StablehloReferenceOps`. The pinned StableHLO reference source constructs a
complex `APFloat` with an older MLIR API; the repository's LLVM revision uses
`mlir::NonFloatComplex<APFloat>`.

Apply this one-line type compatibility patch before building the compiler from
a fresh checkout:

```sh
git -C third_party/stablehlo apply --check ../../compiler/patches/stablehlo-reference-mlir-complex.patch
git -C third_party/stablehlo apply ../../compiler/patches/stablehlo-reference-mlir-complex.patch
```

If the patch is already applied, verify that state instead of applying it again:

```sh
git -C third_party/stablehlo apply --reverse --check ../../compiler/patches/stablehlo-reference-mlir-complex.patch
```

The patch changes the constructor type only. It does not change the reference
algorithm or the submodule revision. Keep unrelated submodule edits intact.
