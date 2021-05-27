from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

baseline = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 3), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

C10_v1 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 1),('skip_connect', 2)], reduce_concat=range(2, 6))
C10_v2 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1),('skip_connect', 2)], reduce_concat=range(2, 6))
C10_v3 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

C10_nw_v1 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1),('max_pool_3x3', 0)], reduce_concat=range(2, 6))
C10_nw_v2 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 1),('skip_connect', 2)], reduce_concat=range(2, 6))
C10_nw_v3 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1),('max_pool_3x3', 0)], reduce_concat=range(2, 6))

C10_100_v1 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2),('max_pool_3x3', 1)], reduce_concat=range(2, 6))
C10_100_v2 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))
C10_100_v3 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
C10_100_v4 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),('max_pool_3x3', 1)], reduce_concat=range(2, 6))

C100_v1 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3),('skip_connect', 2)], reduce_concat=range(2, 6))


noise_v1 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 2),('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
noise_v2 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2),('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
noise_v3 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))
noise_nw_v5 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 3), ('avg_pool_3x3', 1),('sep_conv_5x5', 3)], reduce_concat=range(2, 6))


C10_L2_v1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
C10_L2_v2 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4)],normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3),('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
C10_L2_v3 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1)],normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0),('max_pool_3x3', 1)], reduce_concat=range(2, 6))

C10_L50_v1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
C10_L50_v2 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
C10_L50_v3 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

C10_L100_v1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 1)], reduce_concat=range(2, 6))
C10_L100_v2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
C10_L100_v3 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

C10_L200_v1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))
C10_L200_v2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
C10_L200_v3 = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

C10_L1000_v1 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
C10_L1000_v2 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
C10_L1000_v3 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
C10_RL1000_v4= Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
C10_RL1000_v5= Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))
C10_RL1000_v6= Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))

C10_L2000_v1 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
C10_L2000_v2 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
C10_L2000_v3 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

C10_L5000_v1 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 0),('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2),('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
C10_L5000_v2 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
C10_L5000_v3 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 3), ('max_pool_3x3', 0), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

C10_L10000_v1 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 0), ('avg_pool_3x3', 3), ('skip_connect', 0), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))
C10_L10000_v2 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3), ('skip_connect', 0), ('avg_pool_3x3', 3), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
C10_L10000_v3 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3), ('avg_pool_3x3', 2), ('avg_pool_3x3', 4), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

C10_L1000_noise_weight = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('avg_pool_3x3', 4),('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1),('dil_conv_5x5', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))


C10_L2_noise_v1 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 1), ('sep_conv_5x5', 3), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
C10_L2_noise_v2 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2),('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

C10_L50_noise_v1 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1),('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
C10_L50_noise_v2 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3),('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

C10_L100_noise_v1 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1),('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
C10_L100_noise_v2 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1),('sep_conv_5x5', 3), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

C10_L200_noise_v1 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3),('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))
C10_L200_noise_v2 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3),('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

C10_L1000_noise_v1 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('dil_conv_5x5', 0), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))
C10_L1000_noise_v2 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3),('skip_connect', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
C10_L1000_noise_v3 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
C10_L1000_noise_v4 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('avg_pool_3x3', 4), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

C10_L2000_noise_v1 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 3), ('avg_pool_3x3', 0), ('avg_pool_3x3', 4),('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 0),('dil_conv_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
C10_L2000_noise_v2 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 3), ('avg_pool_3x3', 4), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

C10_L5000_noise_v1 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 0), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2),('sep_conv_5x5', 0), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))
C10_L5000_noise_v2 = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 3), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3),('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

C10_L10000_noise_v1 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 4), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))
C10_L10000_noise_v2 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 1), ('skip_connect', 4), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 3), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))




DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
DARTS = DARTS_V2





C100_L2_generate_v1 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))





C100_L50_generate_v1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))



C100_L100_generate_v1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))




C100_L200_generate_v1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

C100_L1000_generate_v1 =  Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))



C100_L2000_generate_v1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

C100_L5000_generate_v1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))


C100_L10000_generate_v1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))





C100_L2_generate_gaussian_without_norm_v1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))



C100_L50_generate_gaussian_without_norm_v1 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))




C100_L100_generate_gaussian_without_norm_v1 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))




C100_L200_generate_gaussian_without_norm_v1 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))



C100_L1000_generate_gaussian_without_norm_v1 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))



C100_L2000_generate_gaussian_without_norm_v1 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))


C100_L5000_generate_gaussian_without_norm_v1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

C100_L10000_generate_gaussian_without_norm_v1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))


