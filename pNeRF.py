import math
import collections
import jax.numpy as onp


# Constants
NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3
BOND_LENGTHS = onp.array([1.32, 1.53, 1.47], dtype=onp.float32)  
BOND_ANGLES = onp.array([1.9897, 2.0246, 2.1468], dtype=onp.float32)


def dihedral_to_point_n(dihedral, bond_lengths=BOND_LENGTHS,
                      bond_angles=BOND_ANGLES):
    """
    Takes triplets of dihedral angles (phi, psi, omega) and returns 3D points
    ready for use in reconstruction of coordinates. Bond lengths and angles
    are based on idealized averages.
    :param dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    :return: Tensor containing points of the protein's backbone atoms.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """
    num_steps = dihedral.shape[0]
    batch_size = dihedral.shape[1]

    r_cos_theta = onp.array(bond_lengths * onp.cos(onp.pi - bond_angles))
    r_sin_theta = onp.array(bond_lengths * onp.sin(onp.pi - bond_angles))

    point_x = onp.tile(r_cos_theta.reshape(1, 1, -1), (num_steps,batch_size,1))
    point_y = onp.cos(dihedral) * r_sin_theta
    point_z = onp.sin(dihedral) * r_sin_theta

    point = onp.stack([point_x, point_y, point_z])
    point_perm = onp.transpose(point, (1,3,2,0))
    point_perm = onp.ravel(point_perm)                                # contiguous array
    point_final = point_perm.reshape(num_steps*NUM_DIHEDRALS,
                                               batch_size,
                                               NUM_DIMENSIONS)
    
    return point_final



def point_to_coordinate_n(points_n, num_fragments=6):
    """
    Takes points from dihedral_to_point and sequentially converts them into
    coordinates of a 3D structure.
    Reconstruction is done in parallel by independently reconstructing
    num_fragments and the reconstituting the chain at the end in reverse order.
    The core reconstruction algorithm is NeRF, based on
    DOI: 10.1002/jcc.20237 by Parsons et al. 2005.
    The parallelized version is described in
    https://www.biorxiv.org/content/early/2018/08/06/385450.
    :param points: Tensor containing points as returned by `dihedral_to_point`.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]                        
    :param num_fragments: Number of fragments in which the sequence is split
    to perform parallel computation.
    :return: Tensor containing correctly transformed atom coordinates.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """

    # Compute optimal number of fragments if needed
    total_num_angles = points_n.shape[0] # NUM_STEPS x NUM_DIHEDRALS
    
    if num_fragments is None:
        num_fragments = int(math.sqrt(total_num_angles))
    
    # Initial three coordinates (specifically chosen to eliminate need for
    # extraneous matmul)
    Triplet = collections.namedtuple('Triplet', 'a, b, c')
    batch_size = points_n.shape[1]
    init_matrix = onp.array([[-onp.sqrt(1.0 / 2.0), onp.sqrt(3.0 / 2.0), 0],
                         [-onp.sqrt(2.0), 0, 0], [0, 0, 0]],
                        dtype=onp.float32)
    
    init_coords = [onp.tile(row, (num_fragments * batch_size, 1))
                    .reshape(num_fragments, batch_size, NUM_DIMENSIONS) 
                    for row in init_matrix]
    init_coords = Triplet(*init_coords)                                         # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]
    
    # Pad points to yield equal-sized fragments
    padding = ((num_fragments - (total_num_angles % num_fragments))
               % num_fragments)                                                 # (NUM_FRAGS x FRAG_SIZE) - (NUM_STEPS x NUM_DIHEDRALS)
    points_n = onp.pad(points_n, ((0,padding), (0,0), (0,0)), mode='constant')  # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    points_n = points_n.reshape(num_fragments, -1, batch_size, NUM_DIMENSIONS)  # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    points_n = onp.transpose(points_n, (1,0,2,3))                               # [FRAG_SIZE, NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    
    # Extension function used for single atom reconstruction and whole fragment
    # alignment
    def extend(prev_three_coords, point, multi_m):
        """
        Aligns an atom or an entire fragment depending on value of `multi_m`
        with the preceding three atoms.
        :param prev_three_coords: Named tuple storing the last three atom
        coordinates ("a", "b", "c") where "c" is the current end of the
        structure (i.e. closest to the atom/ fragment that will be added now).
        Shape NUM_DIHEDRALS x [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMENSIONS].
        First rank depends on value of `multi_m`.
        :param point: Point describing the atom that is added to the structure.
        Shape [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        First rank depends on value of `multi_m`.
        :param multi_m: If True, a single atom is added to the chain for
        multiple fragments in parallel. If False, an single fragment is added.
        Note the different parameter dimensions.
        :return: Coordinates of the atom/ fragment.
        """
        # Normalize rows: https://necromuralist.github.io/neural_networks/posts/normalizing-with-numpy/    
        Xbc = (prev_three_coords.c - prev_three_coords.b)
        bc = Xbc/onp.linalg.norm(Xbc, axis = -1, keepdims=True)
        
        Xn = onp.cross(prev_three_coords.b - prev_three_coords.a, bc, axisa=-1, axisb=-1, axisc=-1)              
        n = Xn/onp.linalg.norm(Xn, axis= -1, keepdims=True)                                                                                          
        
        
        if multi_m:     # multiple fragments, one atom at a time
            m = onp.transpose(onp.stack([bc, onp.cross(n, bc), n]), (1, 2, 3, 0))
        else:           # single fragment, reconstructed entirely at once.                                
            s = point.shape + (3,) # +
            m = onp.transpose(onp.stack([bc, onp.cross(n, bc), n]), (1, 2, 0))
            m = onp.tile(m, (s[0], 1, 1)).reshape(s)
        
        coord = onp.squeeze(onp.matmul(m, onp.expand_dims(point, axis=3)), axis=3) + prev_three_coords.c

        return coord


    # Loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially
    # generating the coordinates for each fragment across all batches
    coords_list = [None] * points_n.shape[0]                                  # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]
    prev_three_coords = init_coords
    for i in range(points_n.shape[0]):    # Iterate over FRAG_SIZE
        coord = extend(prev_three_coords, points_n[i], True)
        coords_list[i] = coord
        prev_three_coords = Triplet(prev_three_coords.b,
                                    prev_three_coords.c,
                                    coord)
    
    coords_pretrans = onp.transpose(onp.stack(coords_list), (1, 0, 2, 3))

    # Loop backwards over NUM_FRAGS to align the individual fragments. For each
    # next fragment, we transform the fragments we have already iterated over
    # (coords_trans) to be aligned with the next fragment
    coords_trans = coords_pretrans[-1]
    
    for i in reversed(range(coords_pretrans.shape[0]-1)):
        # Transform the fragments that we have already iterated over to be
        # aligned with the next fragment `coords_trans`
        transformed_coords = extend(Triplet(*[di[i]
                                              for di in prev_three_coords]),
                                    coords_trans, False)
        coords_trans = onp.concatenate([coords_pretrans[i], transformed_coords], 0)    
    coords = onp.pad(coords_trans[:total_num_angles-1], ((1, 0), (0, 0), (0, 0)))
    
    
    return coords
