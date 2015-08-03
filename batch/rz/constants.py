QUEUES = ('f_ocean', 'f_ocean2', 'express', 'small', 'medium', 'long', 'para_low')

# NODES_KIND = ('westmere', 'shanghai', 'f_ocean', 'f_ocean2', 'amd256')
# NODES_CPU_POWER = (2.67/4, 2.4/4, 2.1/4, 2.6, 2.1)
# NODES_MAX = (12, 26, 12, 12, 1)
# NODES_LEFT_FREE = (0, 0, 0, 0, 1)

NODE_INFOS = {'westmere': (2.67/4, 18, 0), 'shanghai': (2.4/4, 26, 0), 'f_ocean': (2.1/4, 12, 0), 'f_ocean2': (2.6, 12, 0), 'amd128': (2.4/4, 1, 1), 'amd256': (2.1/4, 1, 1), 'fobigmem': (2.6, 2, 2)}

MAX_WALLTIME = {'express': 3, 'small': 24, 'medium': 240, 'long': 480, 'para_low': 1000}

# Westmere nodes (12 CPUs per node, 2.67 GHz)
# f_ocean Barcelona nodes (8 CPUs per node, 2.1 GHz) (f_ocean queue)
# f-ocean2 nodes Intel(R) Xeon(R) CPU E5-2670 0 (16 CPUs per node, 2.6 GHz) (f_ocean2 queue)
# f_ocean2 express nodes (16 CPUs per node, 2.6 GHz) (foexpress queue)
# Opteron nodes (4 CPUs per node, 2.8 GHz)
# 3 AMD-Shanghai nodes (16 CPUs per node, 2.4 GHz)
# 1 AMD-Magny node (48 CPUs per node, 2.1 GHz)
# Shanghai Ethernet nodes (8 CPUs per node, 2.4 GHz) (bio_ocean queue)
# Shanghai Infiniband nodes (8 CPUs per node, 2.4 GHz) (math queue)

QNODES_COMMAND = '/usr/local/bin/qnodes'
QSUB_COMMAND = '/opt/pbs/default/bin/qsub'
QSTAT_COMMAND = '/opt/pbs/default/bin/qstat'