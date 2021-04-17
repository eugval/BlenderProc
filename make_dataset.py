import subprocess
import glob
import shutil
import os
import time

temp_output = "current_temp_output"
# final_output = "output"
file_counter = 0
start_time = time.time()
print('start_time : {}'.format(start_time))
for i in range(3000):
    print('ITERATION {}'.format(i))

    # os.environ["BLENDER_PROC_RANDOM_SEED"] = str(int(start_time*10000000)+i)
    # os.environ["BLENDER_PROC_RANDOM_SEED"] = str(1232)
    cmd = ["python", "run.py", "examples/simple_coffee_mugs/config.yaml", "examples/simple_coffee_mugs/output" , "resources/ModelNet40/"]
    subprocess.check_output(cmd)

    print("Done in : {}hrs".format((time.time()-start_time)/3600.))

#     for file in glob.glob(f"{temp_output}/*.hdf5"):
#         shutil.move(file, f"{final_output}/{file_counter}.hdf5")
#         file_counter += 1
# shutil.rmtree(temp_output)

