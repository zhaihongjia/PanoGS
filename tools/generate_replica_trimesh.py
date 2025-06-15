import numpy as np
from plyfile import PlyData, PlyElement
import ArgumentParser

def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file."""
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                norms[i, 0],
                norms[i, 1],
                norms[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    for i in range(faces.shape[0]):
        if faces.shape[1] == 3:
            ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))
        elif faces.shape[1] == 4:
            ply_file.write("4 %d %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2], faces[i, 3]))

    ply_file.close()

def generate_trimesh(input_ply_file, output_ply_file):
    ply_data = PlyData.read(input_ply_file)

    quads = [np.array(i).tolist() for i in ply_data.elements[1]]
    quads = np.squeeze(np.stack(quads, 0))

    verts = [np.array(i).tolist()[:3] for i in np.array(ply_data.elements[0])]
    verts = np.stack(verts, 0)
    normls = [np.array(i).tolist()[3:6] for i in np.array(ply_data.elements[0])]
    normls = np.stack(normls, 0) * -1
    colors = [np.array(i).tolist()[6:9] for i in np.array(ply_data.elements[0])]
    colors = np.stack(colors, 0)

    # turn Quad Mesh into Tri Mesh
    faces = []
    for i in range(quads.shape[0]):
        quad = quads[i]  
        faces.append([quad[0], quad[1], quad[2]])  
        faces.append([quad[0], quad[2], quad[3]])  
    
    faces = np.stack(faces, 0)
    print('faces: ', faces.shape)

    meshwrite(output_ply_file, verts, faces, normls, colors)

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate trimesh.")
    parser.add_argument("--replica_dir", type=str)
    args = parser.parse_args()

    scene_ids = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']
    for scene_id in scene_ids:

        input_ply_file = os.path.join(args.replica_dir, '{}_mesh.ply'.format(scene_id))  
        output_ply_file = os.path.join(args.replica_dir, '{}_trimesh.ply'.format(scene_id)) 

        generate_trimesh(input_ply_file, output_ply_file)
        print('save: ', output_ply_file)
