# Asset contracts and official extension examples

This sample is the executable contract reference for Feather's open Asset API. A normal C# project
derives from the five framework bases—`TextureAsset`, `MaterialAsset`, `ModelAsset`, `SceneAsset`,
and `ActorAsset`—and the source generator emits stable nominal inheritance, logical references,
capabilities, inputs, and inherited product requirements into `Generated/asset-manifest.json`.

`PngTextureAsset` and `GltfModelAsset` use the exact type identities consumed by Feather Studio's
real PNG and GLB import providers. `LayeredMaterialAsset`, `SceneDocumentProjectionAsset`, and
`ActorTemplateDefinitionAsset` show cross-domain composition through `AssetRef<T>` without storing
paths, bytes, GPU handles, or live Scene entities in the authoring payload. Here `ModelAsset`
always means an instantiable 3D model; learned models require a separately named contract.

`TextureAtlasAsset` and `SignedDistanceFieldTextureAsset` are deliberately official extensions,
not new core base classes. They inherit the standard Texture View requirement and declare only the
metadata and logical source references a provider needs. A provider can later realise those
products through the same public import/transform/build boundary used by first-party and package
types; adding either declaration requires no new renderer union or GraphCore node kind.

Build the sample from the Feather checkout:

```sh
dotnet build samples/AssetContracts/AssetContracts.csproj
```

The generated manifest is build output and is ignored by Git. Inspect it to see stable Type IDs,
`referencedAssetTypeId` fields, exact UTF-8 source hashes, and the nominal base IDs imported from
the Feather framework manifest.
