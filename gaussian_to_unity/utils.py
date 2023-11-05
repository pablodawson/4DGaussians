import numpy as np
import struct
import os
import json

kScaler = (1 << 21) - 1

VectorFormats = {'Float32':0, 'Norm16':1, 'Norm11':2, 'Norm6':3}

SHFormats = {'Float32':0, 'Float16':1, 'Norm11':2, 'Norm6':3, 
             'Cluster64k':4, 'Cluster32k':5, 'Cluster16k':6, 
             'Cluster8k':7, 'Cluster4k':8}

ColorFormats = {'Float32x4':0, 'Float16x4':1, 'Norm8x4':2, 'BC7':3}

def morton_part1by2(x):
    x &= 0x1fffff
    x = (x ^ (x << 32)) & 0x1f00000000ffff
    x = (x ^ (x << 16)) & 0x1f0000ff0000ff
    x = (x ^ (x << 8))  & 0x100f00f00f00f00f
    x = (x ^ (x << 4))  & 0x10c30c30c30c30c3
    x = (x ^ (x << 2))  & 0x1249249249249249
    return x

def morton_encode3(v):
    return (morton_part1by2(v[2]) << 2) | (morton_part1by2(v[1]) << 1) | morton_part1by2(v[0])

def reorder_morton_job(bounds_min, inv_bounds_size, splat_data):
    n = len(splat_data)
    order = np.zeros((n, 2), dtype=np.uint64)
    splat_data  = splat_data.cpu().numpy()
    pos = (splat_data - bounds_min) * inv_bounds_size * kScaler
    ipos = pos.astype(np.uint32)

    for index in range(n):
        code = morton_encode3(ipos[index])
        order[index] = (code, index)
    
    return np.array(sorted(order, key=lambda element: element[0]))

    #return order

def calculate_bounds(pos):
    mins = np.min(pos, axis=0)
    maxs = np.max(pos, axis=0)
    return mins, maxs

def encode_float3_to_norm16(v):
    x = int(v[0] * 65535.5)
    y = int(v[1] * 65535.5)
    z = int(v[2] * 65535.5)
    return x | (y << 16) | (z << 32)

def encode_float3_to_norm11(v):
    x = int(v[0] * 2047.5)
    y = int(v[1] * 1023.5)
    z = int(v[2] * 2047.5)
    return x | (y << 11) | (z << 21)

def encode_float3_to_norm655(v):
    x = int(v[0] * 63.5)
    y = int(v[1] * 31.5)
    z = int(v[2] * 31.5)
    return x | (y << 6) | (z << 11)

def encode_float3_to_norm565(v):
    x = int(v[0] * 31.5)
    y = int(v[1] * 63.5)
    z = int(v[2] * 31.5)
    return x | (y << 5) | (z << 11)

def encode_quat_to_norm10(v):
    x = int(v[0] * 1023.5)
    y = int(v[1] * 1023.5)
    z = int(v[2] * 1023.5)
    w = int(v[3] * 3.5)
    return x | (y << 10) | (z << 20) | (w << 30)

def encode_vector(v, vector_format):
    v = np.clip(v, 0, 1)
    
    if vector_format == "Float32":
        return struct.pack('fff', v[0], v[1], v[2])
        
    elif vector_format == "Norm16":
        enc = encode_float3_to_norm16(v)
        low_bits = enc & 0xFFFFFFFF
        high_bits = enc >> 32
        return struct.pack('IH', low_bits, high_bits)

    elif vector_format == "Norm11":
        enc = encode_float3_to_norm11(v)
        return struct.pack('I', enc)

    elif vector_format == "Norm6":
        enc = encode_float3_to_norm655(v)
        return struct.pack('H', enc)
    
    elif vector_format == "Norm10":
        enc = encode_quat_to_norm10(v)
        return struct.pack('I', enc)
    
    else:
        raise ValueError(f"Unknown vector format: {vector_format}")

def encode_quat_to_norm10(v):
    return int(v[0] * 1023.5) | (int(v[1] * 1023.5) << 10) | (int(v[2] * 1023.5) << 20) | (int(v[3] * 3.5) << 30)

def normalize_chunk(chunk):
    bounds_min, bounds_max = calculate_bounds(chunk)
    normalized_chunk = (chunk - bounds_min) / (bounds_max - bounds_min + 1e-6)
    return normalized_chunk, bounds_min, bounds_max

def create_chunks(mean3d_in, scale, gaussianCount, chunk_size):
    
    pos_chunks = []
    scale_chunks = []

    positions = np.zeros((gaussianCount, 3), dtype=np.float32)
    scales = np.zeros((gaussianCount, 3), dtype=np.float32)

    for i in range(0, gaussianCount, chunk_size):
        
        chunk_pos = mean3d_in[i:i+chunk_size].copy()
        chunk_scale = scale[i:i+chunk_size].copy()
        
        positions[i:i+chunk_size], min_pos, max_pos = normalize_chunk(chunk_pos)
        scales[i:i+chunk_size], min_scale, max_scale = normalize_chunk(chunk_scale)

        pos_chunks.append([min_pos, max_pos])
        scale_chunks.append([min_scale, max_scale])
        
    return positions, scales, pos_chunks, scale_chunks

def create_positions_asset(means3D_sorted, basepath, format='Norm11', idx=-1, one_file=False):
    if one_file:
        if idx == 0:
            if os.path.exists(os.path.join(basepath, f"position_data.bytes")):
                os.remove(os.path.join(basepath, f"position_data.bytes"))

        path = os.path.join(basepath, f"position_data.bytes")

        with open(path, 'ab') as f:
            for mean3d in means3D_sorted:
                f.write(encode_vector(mean3d, format))
    else:
        output_folder = os.path.join(os.path.dirname(basepath), "positions")
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, f"{idx}.bytes")

        with open(path, 'wb') as f:
            for mean3d in means3D_sorted:
                f.write(encode_vector(mean3d, format))

def create_others_asset(rotations, scales, sh_index, basepath, scale_format, idx=0):

    output_folder = os.path.join(os.path.dirname(basepath), "others")
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"{idx}.bytes")

    with open(path, 'ab') as f:
        for rotation, scale in zip(rotations, scales):
            f.write(encode_vector(rotation, "Norm10"))
            f.write(encode_vector(scale, scale_format))

def f32tof16(f32):
    f16 = np.float16(f32)
    return f16.view(np.uint16)

def create_chunks_asset(pos_chunks, scale_chunks, basepath, idx=0, one_file=False):

    format_str = "ffffffIII"

    if one_file:
        path = os.path.join(basepath, f"chunk_data.bytes")
        if idx == 0:
            if os.path.exists(path):
                os.remove(path)
        mode = 'ab'
    else:
        output_folder = os.path.join(os.path.dirname(basepath), "chunks")
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, f"{idx}.bytes")
        mode = 'wb'
    
    with open(path, mode) as f:
        for pos_chunk, scale_chunk in zip(pos_chunks, scale_chunks):
            # f32 -> f16
            sclX = f32tof16(scale_chunk[0][0]) | (f32tof16(scale_chunk[1][0]) << 16)
            sclY = f32tof16(scale_chunk[0][1]) | (f32tof16(scale_chunk[1][1]) << 16)
            sclZ = f32tof16(scale_chunk[0][2]) | (f32tof16(scale_chunk[1][2]) << 16)

                        
            packed_data = struct.pack(format_str,
                    pos_chunk[0][0], pos_chunk[1][0], pos_chunk[0][1], pos_chunk[1][1], pos_chunk[0][2], pos_chunk[1][2], 
                    sclX, sclY, sclZ)
            
            f.write(packed_data)
    
def normalize_swizzle_rotation(wxyz):
    return np.roll(np.array(wxyz / np.linalg.norm(wxyz)), -1)

# e1 e2 e3 e4
def pack_smallest_3_rotation(q):
    abs_q = np.abs(q)
    index = np.argmax(abs_q, axis=1)
    q_rolled = np.roll(q, -index-1, axis=1)
    signs = np.sign(q_rolled[:, 3])
    three = q_rolled[:, :3] * signs[:, np.newaxis]
    three = (three * np.sqrt(2)) * 0.5 + 0.5
    index = index / 3.0
    return np.column_stack((three, index)) 

def linear_scale(log_scale):
    return np.abs(np.exp(log_scale))

def linealize(rot, scale):
    # Rotation processing
    rot = normalize_swizzle_rotation(rot)
    rot = pack_smallest_3_rotation(rot)

    # Scale processing
    scale = linear_scale(scale)
    scale = scale ** (1.0 / 8.0)

    return rot, scale

def create_one_file(basepath, pos_file_format="Norm11", splat_count=0, chunk_count=0, frame_time=1/20):

    # Current format
    # 1- Metadata
    # 2- Static data
    # 3- Dynamic data, intercalated positions and chunks

    positions_path = os.path.join(basepath, "positions")
    chunks_path = os.path.join(basepath, "chunks")
    
    data = []
    
    # ---- Header information -----

    # For now this is comes from Unity, hence the hardcoding
    format_version = 20231006
    scale_file_format = "Norm11"
    sh_file_format= "Norm6"
    color_format = "Norm8x4"
    color_width = 2048
    color_height = 112

    frame_count = len(os.listdir(positions_path))

    data.append(struct.pack('I', format_version)) # Format version
    data.append(struct.pack('I', splat_count)) # Splat count
    data.append(struct.pack('f', frame_time)) # Frame time
    data.append(struct.pack('I', frame_count)) # Frame count
    data.append(struct.pack('I', chunk_count)) # Chunk count
    data.append(struct.pack('I', VectorFormats[pos_file_format])) # Position format
    data.append(struct.pack('I', VectorFormats[scale_file_format])) # Scale format
    data.append(struct.pack('I', SHFormats[sh_file_format])) # SH format
    data.append(struct.pack('I', ColorFormats[color_format])) # Color format 
    data.append(struct.pack('I', color_width)) # Color width
    data.append(struct.pack('I', color_height)) # Color height

    static_info = ["chunk_static.bytes", "colors.bytes", "others.bytes", "shs.bytes"]
    
    # ---- Static data ----
    
    for info in static_info:
        with open(os.path.join(basepath, info), 'rb') as f:
            size = os.path.getsize(os.path.join(basepath, info))
            data.append(struct.pack('I', size))
            data.append(f.read())
    
    # ---- Dynamic data ----
    
    # Read all the files intercalated
    for position_file in sorted(os.listdir(positions_path)):
        with open(os.path.join(positions_path, position_file), 'rb') as f:
            data.append(f.read())

        with open(os.path.join(chunks_path, position_file), 'rb') as f:
            data.append(f.read())
    
    # Write the data to a single file
    file_name = os.path.join(os.path.dirname(basepath), "scene.bytes")

    with open(file_name, 'wb') as f:
        for chunk in data:
            f.write(chunk)


def create_one_file_chunk_pos(basepath):

    positions_path = os.path.join(os.path.dirname(basepath), "positions")
    chunks_path = os.path.join(os.path.dirname(basepath), "chunks")

    pos_data = []
    chunk_data = []

    for position_file in sorted(os.listdir(positions_path)):
        with open(os.path.join(positions_path, position_file), 'rb') as f1:
            pos_data.append(f1.read())

        with open(os.path.join(chunks_path, position_file), 'rb') as f2:
            chunk_data.append(f2.read())
    
    pos_file_name = os.path.join(os.path.dirname(basepath), "position_data.bytes")
    chunk_file_name = os.path.join(os.path.dirname(basepath), "chunk_data.bytes")

    with open(pos_file_name, 'wb') as f1:
        for chunk in pos_data:
            f1.write(chunk)
    
    with open(chunk_file_name, 'wb') as f2:
        for chunk in chunk_data:
            f2.write(chunk)

def create_json(save_path, splat_count=0, chunk_count=0, pos_format='Norm11', save_interval=1, fps=30, frame_count=0):

    metadata = {}

    metadata["splat_count"] = splat_count
    metadata["chunk_count"] = chunk_count
    metadata["pos_format"] = pos_format
    metadata["frame_time"] = save_interval * 1.0 / fps
    metadata["frame_count"] = frame_count

    with open(os.path.join(save_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f)

if __name__=="__main__":
    print("Testing")
    create_one_file("output/martini/cut_beef_newformat", splat_count=298081)