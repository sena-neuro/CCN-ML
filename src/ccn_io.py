import scipy.io as sio
import mne
import numpy as np

# read mat file only
def read_file(fname = "data.mat"):
	"""
		Reading mat files. Use the scipy library or other library depending on the file type.
		Can be extended later again for all use-cases.   
	"""
	try:
		content = sio.loadmat(fname)
		return content
	except Exception: 
		print("\n    Something went wrong!\n")


def read_mne(fname = "data/human_motion_epochs.mat"):
	"""
	To be extended later in order to include dictionary and data name.
	For more read: https://martinos.org/mne/stable/generated/\
	mne.io.read_raw_fieldtrip.html#mne.io.read_raw_fieldtrip
	"""
	
	try:
		data = mne.io.read_raw_fieldtrip( fname,None,data_name="sample_audvis_meg_raw") 
		# try also mne.io.read_epochs_fieldtrip
	except Exception: 
		print("    Something went wrong with read_mne()!\n")
		
def read_txt(fname = "data/EEG Eye State.txt"):
	"""
		Read dataset from txt file and divide the info into x and y. 
	"""
	
	try: 
		# Loading the dataset
		with open(fname) as f:
		    content = f.readlines()

		content = [x.strip() for x in content] 
		content = [x.split(",") for x in content]

		# Converting list to numpy array
		content = np.array(content, dtype = 'float32')

		# Shuffling the dataset
		random.shuffle(content)
		x = content[:, :-1]
		y = np.array(content[:, -1], dtype = 'int32')
		return x,y
	except Exception: 
		print("    Something went wrong with read_txt()!")
		
def print_mat_nested(d, indent=0, nkeys=0):
    """
        Pretty print nested structures from .mat files.
    """

    try:
        # Subset dictionary to limit keys to print.  Only works on first level
        if nkeys>0:
            d = {k: d[k] for k in d.keys()[:nkeys]}  # Dictionary comprehension: limit to first nkeys keys.

        if isinstance(d, dict):
            for key, value in d.iteritems():         # iteritems loops through key, value pairs
                print('\t' * indent + 'Key: ' + str(key))
                print_mat_nested(value, indent+1)

        if isinstance(d,np.ndarray) and d.dtype.names is not None:  # Note: and short-circuits by default
            for n in d.dtype.names:    # This means it's a struct, it's bit of a kludge test.
                print('\t' * indent + 'Field: ' + str(n))
                print_mat_nested(d[n], indent+1)
        print("    Finished Printing!")
    except Exception:
        print("    Something went wrong with print_mat_nested()!")


#------------------------------------
# testing
if __name__ == "__main__":
	print("Trying read_file()!\n")
	content    = read_file("data/human_motion_epochs.mat")
	if content != None:
		print("    read_file() works.\n")
		
	print("Trying read_mne()!\n")
	content    = read_mne("data/sample_audvis_meg_raw.fif")
	if content != None:
		print("    read_mne() works.\n")
		print("Trying read_mne()!\n")
		
	print("Trying read_txt()!\n")
	content    = read_txt()
	if content != None:
		print("    read_txt() works.\n")
		
	print("Trying print_mat_nested()!\n")
	content    = print_mat_nested("data/human_motion_epochs.mat")
	if content != None:
		print("    print_mat_nested() works.\n")

