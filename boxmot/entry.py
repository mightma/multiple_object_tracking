import os
import json
import socket


if __name__ == "__main__":

    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))

    # Parse the IP address of the master node in the multiple nodes cluster of SageMaker training.
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)

    os.environ['DS_BUILD_FUSED_ADAM'] = '1'
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

    # backend env config
    os.environ['FI_PROVIDER'] = 'efa'
    os.environ['NCCL_PROTO'] = 'simple'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['HCCL_OVER_OFI'] = '1'

    # invoke the torch launcher shell script.
    os.system("chmod +x ./run_train.sh")
    os.system("/bin/bash -c ./run_train.sh")
