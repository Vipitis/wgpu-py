"""
based on https://github.com/webgpu/webgpu-samples/tree/main/sample/fractalCube
to show how textures can be used as a target and resource in the same pass without copying.
"""

# test_example = false

import time

import wgpu
import numpy as np

from wgpu.gui.auto import WgpuCanvas, run

# constants
cube_vertex_size = 4 * 10
cube_position_offset = 0
cube_color_offset = 4 * 4
cube_uv_offset = 4 * 8
cube_vertex_count = 36
RESOLUTION = (640, 480)

cube_vertex_array = np.array(
    # float4 position, float4 color, float2 uv
    [
        1, -1, 1, 1,   1, 0, 1, 1,  0, 1,
        -1, -1, 1, 1,  0, 0, 1, 1,  1, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,  1, 0,
        1, -1, -1, 1,  1, 0, 0, 1,  0, 0,
        1, -1, 1, 1,   1, 0, 1, 1,  0, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,  1, 0,

        1, 1, 1, 1,    1, 1, 1, 1,  0, 1,
        1, -1, 1, 1,   1, 0, 1, 1,  1, 1,
        1, -1, -1, 1,  1, 0, 0, 1,  1, 0,
        1, 1, -1, 1,   1, 1, 0, 1,  0, 0,
        1, 1, 1, 1,    1, 1, 1, 1,  0, 1,
        1, -1, -1, 1,  1, 0, 0, 1,  1, 0,

        -1, 1, 1, 1,   0, 1, 1, 1,  0, 1,
        1, 1, 1, 1,    1, 1, 1, 1,  1, 1,
        1, 1, -1, 1,   1, 1, 0, 1,  1, 0,
        -1, 1, -1, 1,  0, 1, 0, 1,  0, 0,
        -1, 1, 1, 1,   0, 1, 1, 1,  0, 1,
        1, 1, -1, 1,   1, 1, 0, 1,  1, 0,

        -1, -1, 1, 1,  0, 0, 1, 1,  0, 1,
        -1, 1, 1, 1,   0, 1, 1, 1,  1, 1,
        -1, 1, -1, 1,  0, 1, 0, 1,  1, 0,
        -1, -1, -1, 1, 0, 0, 0, 1,  0, 0,
        -1, -1, 1, 1,  0, 0, 1, 1,  0, 1,
        -1, 1, -1, 1,  0, 1, 0, 1,  1, 0,

        1, 1, 1, 1,    1, 1, 1, 1,  0, 1,
        -1, 1, 1, 1,   0, 1, 1, 1,  1, 1,
        -1, -1, 1, 1,  0, 0, 1, 1,  1, 0,
        -1, -1, 1, 1,  0, 0, 1, 1,  1, 0,
        1, -1, 1, 1,   1, 0, 1, 1,  0, 0,
        1, 1, 1, 1,    1, 1, 1, 1,  0, 1,

        1, -1, -1, 1,  1, 0, 0, 1,  0, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,  1, 1,
        -1, 1, -1, 1,  0, 1, 0, 1,  1, 0,
        1, 1, -1, 1,   1, 1, 0, 1,  0, 0,
        1, -1, -1, 1,  1, 0, 0, 1,  0, 1,
        -1, 1, -1, 1,  0, 1, 0, 1,  1, 0,
    ],
    dtype=np.float32
)

vertex_code = """
struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) fragPosition: vec4f,
}

@vertex
fn main(
  @location(0) position : vec4f,
  @location(1) uv : vec2f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = uniforms.modelViewProjectionMatrix * position;
  output.fragUV = uv;
  output.fragPosition = 0.5 * (position + vec4(1.0, 1.0, 1.0, 1.0));
  return output;
}
"""

fragment_code = """
@binding(1) @group(0) var mySampler: sampler;
@binding(2) @group(0) var myTexture: texture_2d<f32>;

@fragment
fn main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f
) -> @location(0) vec4f {
  let texColor = textureSample(myTexture, mySampler, fragUV * 0.8 + vec2(0.1));
  let f = select(1.0, 0.0, length(texColor.rgb - vec3(0.5)) < 0.01);
  return f * texColor + (1.0 - f) * fragPosition;
}
"""

# setup context
canvas = WgpuCanvas(size=RESOLUTION, title="wgpu recursive cube example")
adapter:wgpu.GPUAdapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device:wgpu.GPUDevice = adapter.request_device_sync(required_limits=None)
context = canvas.get_context("wgpu")

render_texture_format = context.get_preferred_format(device.adapter)
context.configure(
    device=device,
    format=render_texture_format,
    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.TEXTURE_BINDING,
    )

# create vertex buffer
vertex_buffer = device.create_buffer_with_data(data=cube_vertex_array, usage=wgpu.BufferUsage.VERTEX)

pipeline = device.create_render_pipeline(
    layout= "auto",
    vertex={
        "module": device.create_shader_module(code=vertex_code),
        "entry_point":"main",
        "buffers": [
            {   "array_stride": cube_vertex_size,
                "attributes": [
                    {
                        "shader_location": 0,
                        "offset": cube_position_offset,
                        "format": wgpu.VertexFormat.float32x4,
                    },
                    {
                        "shader_location": 1,
                        "offset": cube_uv_offset,
                        "format": wgpu.VertexFormat.float32x2,
                    },
                ],
            },
        ]
    },
    fragment={
        "module": device.create_shader_module(code=fragment_code),
        "entry_point": "main",
        "targets": [
            {
                "format": render_texture_format,
            }
        ],
    },
    primitive= {
        "topology": wgpu.PrimitiveTopology.triangle_list,
        "cull_mode": wgpu.CullMode.none,
    },

    depth_stencil={
        "depth_write_enabled": True,
        "depth_compare": wgpu.CompareFunction.less,
        "format": wgpu.TextureFormat.depth24plus,
    }
)

depth_texture = device.create_texture(
    size=RESOLUTION,
    format=wgpu.TextureFormat.depth24plus,
    usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
)

uniform_buffer = device.create_buffer(
    size=4*16,
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

# TODO: try to avoid this temporal texture?
cube_texture = device.create_texture(
    size=RESOLUTION,
    format=render_texture_format,
    usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
)

sampler = device.create_sampler(
    mag_filter=wgpu.FilterMode.linear,
    min_filter=wgpu.FilterMode.linear,
)

uniform_bind_group = device.create_bind_group(
    layout=pipeline.get_bind_group_layout(0),
    entries=[
        {
            "binding": 0,
            "resource": {
                "buffer": uniform_buffer,
            },
        },
        {
            "binding": 1,
            "resource": sampler,
        },
        {
            "binding": 2,
            "resource": cube_texture.create_view(), #TODO: this is the one we want to replace by current_texture.create_view(usage=wgpu.TextureUsage.TEXTURE_BINDING)
        },
    ],
)

render_pass_descriptor = {
    "color_attachments": [
        {
            "view": None, # assigned later (in the loop?)
            "clear_value": (0.5, 0.5, 0.5, 1.0),
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
        }
    ],
    "depth_stencil_attachment": {
        "view": depth_texture.create_view(),
        "depth_clear_value": 1.0,
        "depth_load_op": wgpu.LoadOp.clear,
        "depth_store_op": wgpu.StoreOp.store,
    }
}

aspect = RESOLUTION[0] / RESOLUTION[1]
f = np.tan(np.pi * 0.5 - 0.5 * ((2 * np.pi) / 5))
projection_matrix = np.array([
    f, 0.0, 0.0, 0.0,
    0.0, f, 0.0, 0.0,
    0.0, 0.0, -1.0101009607315063, -1, # those are approximate values... hope still works?
    0.0, 0.0, -1.0101009607315063, 0.0,
], dtype=np.float32).reshape(4, 4)
model_view_projection_matrix = np.zeros((4, 4), dtype=np.float32)

def _rotate_matrix(matrix, axis, angle, dst):
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    # Rotation matrix elements
    r00 = x*x + (1-x*x)*c
    r01 = x*y*t - z*s
    r02 = x*z*t + y*s
    r10 = x*y*t + z*s
    r11 = y*y + (1-y*y)*c
    r12 = y*z*t - x*s
    r20 = x*z*t - y*s
    r21 = y*z*t + x*s
    r22 = z*z + (1-z*z)*c
    # Build rotation matrix
    R = np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ], dtype=np.float32)
    dst[:3,:4] = R @ matrix[:3,:4]
    dst[3,:] = matrix[3,:]
    return dst

def get_transformation_matrix(seconds):
    view_matrix = np.identity(4, dtype=np.float32)
    view_matrix[3, 2] = -4
    rotation_axis = np.array([np.sin(seconds), np.cos(seconds), 0.0], dtype=np.float32)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    view_matrix = _rotate_matrix(view_matrix, rotation_axis, 1, view_matrix)
    res = np.matmul(projection_matrix.T, view_matrix.T)
    res = np.ascontiguousarray(res.T)
    return res


def draw_frame():
    now_time = time.time()
    transformation_matrix = get_transformation_matrix(now_time)
    device.queue.write_buffer(uniform_buffer, 0, transformation_matrix)

    swap_chain_texture = context.get_current_texture()
    render_pass_descriptor["color_attachments"][0]["view"] = swap_chain_texture.create_view(usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

    command_encoder:wgpu.GPUCommandEncoder = device.create_command_encoder()
    pass_encoder = command_encoder.begin_render_pass(**render_pass_descriptor)
    pass_encoder.set_pipeline(pipeline)
    pass_encoder.set_bind_group(0, uniform_bind_group)
    pass_encoder.set_vertex_buffer(0, vertex_buffer)
    pass_encoder.draw(vertex_count=cube_vertex_count)
    pass_encoder.end()

    # TODO: the copy we want to avoid
    command_encoder.copy_texture_to_texture(
        {
            "texture": swap_chain_texture,
        },
        {
            "texture": cube_texture,
        },
        RESOLUTION,
    )

    device.queue.submit([command_encoder.finish()])
    canvas.request_draw()

canvas.request_draw(draw_frame)

if __name__ == "__main__":
    run()

