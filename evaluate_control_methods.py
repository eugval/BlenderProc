import subprocess
import glob
import shutil
import os
import time
import psutil
import json

print('PID {}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(os.getpid()))
def is_running(pid=None):
    if(pid is not None):
        return psutil.pid_exists(pid)
    return False

def wait_for_pid(pid, wait_for=1800):
    while is_running(pid):
      print('process is running, waiting for {:.1f}min'.format(wait_for/60.))
      time.sleep(wait_for)

id_to_wait = 10346 # 2222 #15960  #15579 # 13962 # 12215 # None
wait_for_pid(id_to_wait)


controller_types = [ 'regression'] #'segmentation', 'segmentation_rgb'   'regression', 'regression_noise'
environment_types = ['extra_1'] # , 'perlin',  'lights', 'extra_1', 'extra_2', 'extra_3', 'random' perfect
for controller_type in controller_types:
    for environment_type in environment_types:
        done = False
        i = 0
        evaluations = 100

        while not done:


            print('ITERATION {}'.format(i))

            os.environ["BLENDER_PROC_RANDOM_SEED"] = str(i)

            cmd = ["python",
                   "run.py",
                   "examples/visual_servo/controller_sim_testing.py",
                   "{}".format(controller_type),
                   "{}".format(environment_type),
                   "{}".format(i),
                   "{}".format(1),
                   "--debug_py"
                   ]

            try:
                subprocess.check_output(cmd)
            except subprocess.CalledProcessError as e:
                print('ERROR IN SUBPROCESS, SKIPPING !!!!!!!!')
                evaluations += 1
                if e.output.startswith(b'error: {'):
                    error = json.loads(e.output[7:])  # Skip "error: "
                    print(error['code'])
                    print(error['message'])


            i+=1
            if(i==evaluations):
                done = True