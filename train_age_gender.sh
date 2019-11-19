#!/bin/bash
#SBATCH --job-name=mbm_age
#SBATCH -t 6-00:00                    # tiempo maximo en el cluster (D-HH:MM)
#SBATCH --gres=gpu
#SBATCH -o age_gender.out                 # STDOUT (A = )
#SBATCH -e age_gender.err                 # STDERR
#SBATCH --mail-type=END,FAIL         # notificacion cuando el trabajo termine o falle
#SBATCH --mail-user=afraymon@uc.cl    # mail donde mandar las notificaciones
#SBATCH --workdir=/user/araymond    # direccion del directorio de trabajo
#SBATCH --partition=ialab-high
#SBATCH --nodelist hydra            # forzamos scylla
#SBATCH --nodes 1                    # numero de nodos a usar
#SBATCH --ntasks-per-node=1          # numero de trabajos (procesos) por nodo
#SBATCH --cpus-per-task=1            # numero de cpus (threads) por trabajo (proceso)


source p3/bin/activate
cd mbm/age_gender_detection
python main.py --name $name --cfg $config_file

echo "Finished with job $SLURM_JOBID"
