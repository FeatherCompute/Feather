# Luisa Compute Dynamic Tile Storage Proposal

Feather's compute rasterizer uses a device count pass, prefix pass, and fill
pass to build per-tile primitive references. The pinned Luisa Compute runtime
still requires the backing storage size before dispatch:

* `LuisaCompute/include/luisa/runtime/device.h:234` exposes
  `Device::create_byte_buffer(size_t byte_size)`.
* `LuisaCompute/include/luisa/runtime/dispatch_buffer.h:27-48` stores a host
  `_capacity` and creates the indirect dispatch buffer with that capacity.
* `LuisaCompute/include/luisa/runtime/shader.h:168-172` accepts an existing
  indirect buffer and changes dispatch size, but does not allocate storage.

The current Feather workaround allocates
`triangle_count * kInitialTileReferencesPerTriangle` slots up front. Counts,
prefixes, masks, and overflow detection remain device-side; a frame never
reads the count back to the host. This avoids a synchronization round trip but
can reserve substantially more memory than the visible references require.

## Proposed API

Names are intentionally illustrative for upstream discussion:

```cpp
DynamicBuffer Device::create_dynamic_buffer(
    ByteBuffer element_count,
    size_t element_stride,
    size_t capacity_hint = 0u) noexcept;

DynamicDispatchToken ShaderInvoke::dispatch_indirect_alloc(
    const DynamicBuffer &buffer,
    uint32_t count_offset = 0u,
    uint32_t max_count = std::numeric_limits<uint32_t>::max()) && noexcept;
```

The token must be consumable by a subsequent dispatch in the same stream. The
runtime must expose a device-visible overflow status and define behavior when
`max_count` is exceeded; silent truncation is not acceptable. Vulkan can back
the object with an allocator-managed storage arena and device-side offsets.
Metal can use an argument-buffer suballocation, with a documented conservative
fallback when the device cannot grow an arena.

This API would let Feather replace the fixed reference/mask reservations with
an exact count-to-allocation flow while preserving the existing no-host-
roundtrip contract. It is not implementable as a Feather-only wrapper around
the pinned public API because allocation and binding occur before the count
dispatch completes.
