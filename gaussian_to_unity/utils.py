import numpy as np
import struct
import os

kScaler = (1 << 21) - 1
Formats = {'Float32':12, 'Norm16':6, 'Norm11':4, 'Norm6':2}

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

def create_chunks(mean3d, scale, gaussianCount, chunk_size):
    
    pos_chunks = []
    scale_chunks = []

    positions = np.zeros((gaussianCount, 3), dtype=np.float32)
    scales = np.zeros((gaussianCount, 3), dtype=np.float32)

    for i in range(0, gaussianCount, chunk_size):
        
        chunk_pos = mean3d[i:i+chunk_size].copy()
        chunk_scale = scale[i:i+chunk_size].copy()
        
        positions[i:i+chunk_size], min_pos, max_pos = normalize_chunk(chunk_pos)
        scales[i:i+chunk_size], min_scale, max_scale = normalize_chunk(chunk_scale)

        pos_chunks.append([min_pos, max_pos])
        scale_chunks.append([min_scale, max_scale])
        
    return positions, scales, pos_chunks, scale_chunks

def create_positions_asset(means3D_sorted, basepath, format='Norm11', idx=0):
    
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

def create_chunks_asset(pos_chunks, scale_chunks, basepath, idx=0):

    output_folder = os.path.join(os.path.dirname(basepath), "chunks")
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"{idx}.bytes")
    format_str = "ffffffIII"
    
    with open(path, 'ab') as f:
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

def create_one_file(basepath, pos_file_format="Norm11"):
    positions_path = os.path.join(os.path.dirname(basepath), "positions")
    chunks_path = os.path.join(os.path.dirname(basepath), "chunks")
    
    data = []
    
    # Header information
    frame_count = len(os.listdir(positions_path))
    data.append(struct.pack('I', frame_count))
    data.append(struct.pack('I', Formats[pos_file_format]))
    
    # Read all the files intercalated
    for position_file in sorted(os.listdir(positions_path)):
        with open(os.path.join(positions_path, position_file), 'rb') as f:
            data.append(f.read())

        with open(os.path.join(chunks_path, position_file), 'rb') as f:
            data.append(f.read())
    
    # Write the data to a single file
    file_name = os.path.join(os.path.dirname(basepath), "dynamic_data.bytes")

    with open(file_name, 'wb') as f:
        for chunk in data:
            f.write(chunk)

if __name__=="__main__":
    create_one_file("output/cookie/render_unity_neww")