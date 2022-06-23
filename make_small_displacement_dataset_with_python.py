import subprocess
import glob
import shutil
import os
import time
import psutil
import json

def is_running(pid=None):
    if(pid is not None):
        return psutil.pid_exists(pid)
    return False

def wait_for_pid(pid, wait_for=1800):
    while is_running(pid):
      print('process is running, waiting for {:.1f}min'.format(wait_for/60.))
      time.sleep(wait_for)


id_to_wait = None

wait_for_pid(id_to_wait)

print('PID {}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(os.getpid()))

temp_output = "current_temp_output"
# final_output = "output"
file_counter = 0
start_time = time.time()
print('start_time : {}'.format(start_time))
for i in range(8000):
    print('ITERATION {}'.format(i))

    # os.environ["BLENDER_PROC_RANDOM_SEED"] = str(int(start_time*10000000)+i)
    # os.environ["BLENDER_PROC_RANDOM_SEED"] = str(1232)

    cmd = ["python", "run.py"
        ,"examples/simple_coffee_mugs/distractors_ablation.py"]

    try:
        subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        print('ERROR IN SUBPROCESS, SKIPPING !!!!!!!!')
        if e.output.startswith(b'error: {'):
            error = json.loads(e.output[7:])  # Skip "error: "
            print(error['code'])
            print(error['message'])

    print("Done in : {}hrs".format((time.time()-start_time)/3600.))

#     for file in glob.glob(f"{temp_output}/*.hdf5"):
#         shutil.move(file, f"{final_output}/{file_counter}.hdf5")
#         file_counter += 1
# shutil.rmtree(temp_output)

