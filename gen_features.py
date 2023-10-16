import numpy as np
import os 
from toponetx.classes.simplicial_complex import SimplicialComplex
import trimesh
import random
import glob
import argparse
import os
from tqdm import tqdm
import pickle


def compute_edge_length_squared(x_a, x_b):
    diff = x_a - x_b
    dist = np.sum(diff * diff)
    return dist

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))




def compute_edge_features(cmplex,__mesh):
    hl_to_trimesh_face_id={}
    face_adjacency = {}
    faces = []
    for f in __mesh.faces:
        combinatorial_f = sorted(f)
        faces.append(combinatorial_f)
        hl_to_trimesh_face_id[tuple(combinatorial_f)] = len(hl_to_trimesh_face_id)
        for j in range(0, len(combinatorial_f)):
            edge_key = tuple(sorted([combinatorial_f[j], combinatorial_f[(j + 1) % len(combinatorial_f)]]))
            if edge_key not in face_adjacency:
                face_adjacency[edge_key] = [combinatorial_f]
            else:
                face_adjacency[edge_key].append(combinatorial_f)

        
    edge_features = {}
    edge_to_simplex_id ={}
    for e in cmplex.skeleton(1):
        edge_key = tuple(sorted(e))
        edge_to_simplex_id[edge_key] = len(edge_to_simplex_id)
        edge_features[edge_key] = []

    for i, f_adj in enumerate(__mesh.face_adjacency):
        f = __mesh.faces[f_adj]
        f1 = set(f[0])
        f2 = set(f[1])
        f1_key = tuple(sorted(f1))
        f2_key = tuple(sorted(f2))
        edge_key = tuple(sorted(list(f1.intersection(f2))))
        vid_unshared = sorted(list(f1.symmetric_difference(f2)))
        dh_angle = __mesh.face_adjacency_angles[i]

        area_ABC = __mesh.area_faces[hl_to_trimesh_face_id[f1_key]]
        area_ABD = __mesh.area_faces[hl_to_trimesh_face_id[f2_key]]
        v_a = __mesh.vertices[edge_key[0]]
        v_b = __mesh.vertices[edge_key[1]]
        if vid_unshared[0] not in f1:
            vid_unshared[0], vid_unshared[1] = vid_unshared[1], vid_unshared[0]

        v_c = __mesh.vertices[vid_unshared[0]]
        v_d = __mesh.vertices[vid_unshared[1]]

        e_ab = compute_edge_length_squared(v_a, v_b)
        e_bc = compute_edge_length_squared(v_b, v_c)
        e_ca = compute_edge_length_squared(v_c, v_a)
        e_bd = compute_edge_length_squared(v_b, v_d)
        e_ad = compute_edge_length_squared(v_a, v_d)

        CA = (v_a - v_c)
        CB = (v_b - v_c)

        DA = (v_a - v_d)
        DB = (v_b - v_d)

        angle1 = angle_between(CA, CB)

        angle2 = angle_between(DA, DB)
        if area_ABC==0:
            edge_ratio_ab_1=0
            edge_ratio_ab_2=0
            edge_ratio_bc=0
            edge_ratio_ab_2=0
            edge_ratio_bd=0
            edge_ratio_ad=0
        else:    
            edge_ratio_ab_1 = e_ab / (2. * area_ABC)
            
            edge_ratio_bc = e_bc / (2. * area_ABC)
            edge_ratio_ca = e_ca / (2. * area_ABC)
            edge_ratio_ab_2 = e_ab / (2. * area_ABD)
            edge_ratio_bd = e_bd / (2. * area_ABD)
            edge_ratio_ad = e_ad / (2. * area_ABD)
        edge_span = __mesh.face_adjacency_span[i]
        aaa = [angle1, angle2]
        bbb = [edge_ratio_ab_1, edge_ratio_ab_2]

        edge_features[edge_key] = [dh_angle,
                                   edge_span] + aaa + bbb + [
                                      edge_ratio_bc,
                                      edge_ratio_ca,
                                      edge_ratio_bd,
                                      edge_ratio_ad]
        assert len(edge_features[edge_key]) == 10

    x_e = []
    for edge_key in edge_features:
        e_f = np.array(edge_features[edge_key])
        if len(e_f) != 10:
            f_adj = face_adjacency[edge_key]
            assert len(f_adj) == 1
            f_key = tuple(sorted(f_adj[0]))
            f = set(f_key)
            area_ABC = __mesh.area_faces[hl_to_trimesh_face_id[f_key]]
            area_ABD = area_ABC
            v_a = __mesh.vertices[edge_key[0]]
            v_b = __mesh.vertices[edge_key[1]]
            vid_unshared = sorted(list(f.symmetric_difference(edge_key)))

            v_c = __mesh.vertices[vid_unshared[0]]
            v_d = v_c

            e_ab = compute_edge_length_squared(v_a, v_b)
            e_bc = compute_edge_length_squared(v_b, v_c)
            e_ca = compute_edge_length_squared(v_c, v_a)
            e_bd = compute_edge_length_squared(v_b, v_d)
            e_ad = compute_edge_length_squared(v_a, v_d)

            CA = (v_a - v_c)
            CB = (v_b - v_c)

            DA = (v_a - v_d)
            DB = (v_b - v_d)

            angle1 = angle_between(CA, CB)

            angle2 = angle_between(DA, DB)
            if area_ABC==0:
                edge_ratio_ab_1=0
                edge_ratio_ab_2=0
                edge_ratio_bc=0
                edge_ratio_ab_2=0
                edge_ratio_bd=0
                edge_ratio_ad=0
            else:    
                edge_ratio_ab_1 = e_ab / (2. * area_ABC)
                
                edge_ratio_bc = e_bc / (2. * area_ABC)
                edge_ratio_ca = e_ca / (2. * area_ABC)
                edge_ratio_ab_2 = e_ab / (2. * area_ABD)
                edge_ratio_bd = e_bd / (2. * area_ABD)
                edge_ratio_ad = e_ad / (2. * area_ABD)
            edge_span = e_ab
            aaa = [angle1, angle2]
            bbb = [edge_ratio_ab_1, edge_ratio_ab_2]

            edge_features[edge_key] = [dh_angle,
                                    edge_span] + aaa + bbb + [
                                        edge_ratio_bc,
                                        edge_ratio_ca,
                                        edge_ratio_bd,
                                        edge_ratio_ad]
            x_e.append(np.array(edge_features[edge_key]))
        else:
            x_e.append(e_f)

    # e_f = [edge_features[edge_key] for edge_key in edge_features]
    x_e = np.array(x_e)
    x_e =  np.nan_to_num(x_e, nan=0.0, posinf=0.0, neginf=0.0)
    assert not np.isnan(x_e).any()
    return x_e



def compute_vertex_feature(cmplex,__mesh):

    node_id = list(list(i)[0] for i in cmplex.skeleton(0))

    v_pos = [__mesh.vertices[v] for v in node_id]
    v_normals = [__mesh.vertex_normals[v] for v in node_id]
    v_pos_t = np.array(v_pos)
    v_normals = np.array(v_normals)
    v_f = np.concatenate([v_pos_t, v_normals], axis=1)
    v_f =  np.nan_to_num(v_f, nan=0.0, posinf=0.0, neginf=0.0)
    return v_f

def compute_face_features(cmplex,__mesh, face_2_id):
    f_area = [__mesh.area_faces[face_2_id[tuple(sorted(list(face_from_hl)))]] for face_from_hl in
              cmplex.skeleton(2)]

    f_area = np.array(f_area)
    f_area = f_area / f_area.sum()
    face_normals = [__mesh.face_normals[face_2_id[tuple(sorted(list(face_from_hl)))]] for face_from_hl in
                    cmplex.skeleton(2)]
    f_n = np.array(face_normals)
    face_angles = [__mesh.face_angles[face_2_id[tuple(sorted(list(face_from_hl)))]] for face_from_hl in
                   cmplex.skeleton(2)]
    face_angles = np.array(face_angles)
    t_f = np.concatenate([f_area.reshape(-1, 1), f_n, face_angles], axis=1)
    t_f =  np.nan_to_num(t_f, nan=0.0, posinf=0.0, neginf=0.0)
    return t_f

def compute_face_labels(cmplex,d):
    labeled_face = [ d[ tuple(sorted(list(face)))] for face in
                    cmplex.skeleton(2)]
    return np.array(labeled_face)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', type=str, default="coseg_aliens",
                        help='Data name, can be one of [coseg_aliens, coseg_chairs, coseg_vases]', choices=['coseg_aliens', 'coseg_vases', 'coseg_chairs'])
    
    args = parser.parse_args()
    meshes = f"./{args.name}/"
    label_pth = f"{meshes}" + "sseg/"
    data_name = args.name
    print(data_name)   


    
    # files_labels=glob.glob(labels+"*")
    splits  = ["train","test"]

    training_complex = []
    features_nodes = []
    features_edges = []
    features_faces = []
    labels_= []

    for split in splits:
        files_mesh=glob.glob(f"{meshes}/{split}/*")
        with tqdm(total=len(files_mesh)) as pbar:
            for f in files_mesh:
                base_name=os.path.basename(f)
                label_file =   label_pth + base_name.split('.')[0]+".seseg" 
                # print(label_file)
                labelfile= np.loadtxt(label_file)
                labelfile = labelfile.argmax(axis=1)
                sc = SimplicialComplex.load_mesh(f)
                # sc = tnx.classes.
                mesh = trimesh.load(f,process=False, force=None)
                nodes = mesh.vertices
                faces = mesh.faces
                training_complex.append(sc)
                face_2_id={}
                for i in range(0, len(faces)):
                    face_2_id[tuple(sorted(faces[i]))] = i

                face_features= compute_face_features(sc,mesh,face_2_id)
                
                features_nodes.append(compute_vertex_feature(sc,mesh))
                features_edges.append(compute_edge_features(sc,mesh))
                features_faces.append(face_features)

                faces = [ tuple(sorted(ff)) for ff in faces ]    
                d_label = dict(zip(faces,labelfile))    
                
                labled_faces = compute_face_labels(sc,d_label)
                labels_.append(labled_faces)
                pbar.update(1) 

            with open(f'{data_name}_{split}.pickle', 'wb') as handle:
                pickle.dump({"complexes":training_complex , 
                            "face_label": labels_,
                            "node_feat":features_nodes,
                            "edge_feat":features_edges,
                            "face_feat":features_faces}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
              

        

        #data=np.load("coseg_alien.npz",allow_pickle=True) 

        # data=np.load("coseg_alien.npz",allow_pickle=True)  


        # np.savez( "coseg_alien_.npz",
        #                 **{ "complexes":data["complexes"] , 
        #                 "face_label": np.array(data["face_label"]),
        #                 "node_feat":data["node_feat"],
        #                 "edge_feat":np.array(data["edge_feat"]),
        #                 "face_feat":np.array(data["face_feat"])})   

