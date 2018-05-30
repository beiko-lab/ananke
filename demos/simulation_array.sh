#!/bin/sh
# Give it 4 hours. That should be sufficient for one replication.
#$ -l h_rt=4:00:00
#$ -t 2-47
#$ -cwd
# Needed for Mike's system. Put your environment prep here.
# Let me know if you want an ananke pip/conda environment file.
#export PATH=$PATH:/share/volatile_scratch/mwhall/miniconda3/bin
#source activate ananke
#We need this on our system because the drives are NFS-mounted
#You can probably safely leave this
export HDF5_USE_FILE_LOCKING=FALSE
SIMFILE=simulation_list.csv
SIMULATIONID=$(awk "(NR==$SGE_TASK_ID){print}" $SIMFILE | cut -d , -f 1)
NCLUST=$(awk "(NR==$SGE_TASK_ID){print}" $SIMFILE | cut -d , -f 2)
NTSPERCLUST=$(awk "(NR==$SGE_TASK_ID){print}" $SIMFILE | cut -d , -f 3)
NTP=$(awk "(NR==$SGE_TASK_ID){print}" $SIMFILE | cut -d , -f 4)
NSR=$(awk "(NR==$SGE_TASK_ID){print}" $SIMFILE | cut -d , -f 5)
SIGNALVAR=$(awk "(NR==$SGE_TASK_ID){print}" $SIMFILE | cut -d , -f 6)
SHIFTAMT=$(awk "(NR==$SGE_TASK_ID){print}" $SIMFILE | cut -d , -f 7)

mkdir -p simulations
ananke initialize shape -n ${NTP} -o simulations/simulation_${SIMULATIONID}.h5
ananke import simulation -i simulations/simulation_${SIMULATIONID}.h5 -c ${NCLUST} -t ${NTSPERCLUST} -n ${NSR} -s ${SHIFTAMT} -v ${SIGNALVAR} -o simulations/${SIMULATIONID}
ananke compute_distances -i simulations/simulation_${SIMULATIONID}.h5 -d sts -M 2 -m 0.01 -s 0.01 -z simulations/${SIMULATIONID}_signals.gz -o scores.txt
ananke compute_distances -i simulations/simulation_${SIMULATIONID}.h5 -d euclidean -M 2 -m 0.01 -s 0.01 -z simulations/${SIMULATIONID}_signals.gz -o scores.txt
ananke compute_distances -i simulations/simulation_${SIMULATIONID}.h5 -d dtw -M 2 -m 0.01 -s 0.01 -z simulations/${SIMULATIONID}_signals.gz -o scores.txt
ananke compute_distances -i simulations/simulation_${SIMULATIONID}.h5 -d ddtw -M 2 -m 0.01 -s 0.01 -z simulations/${SIMULATIONID}_signals.gz -o scores.txt
