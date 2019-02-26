import tensorflow as tf
import numpy as np
import time
import sys
import thread
import threading
import RobotRaconteur as RR

RRN = RR.RobotRaconteurNode.s

class tfObject(object):
	def __init__(self):
		 print "Create tensorflow object"
		 self._lock = threading.RLock()
		 self.tf_graph_init()
			
	def tf_graph_init(self):
		with self._lock:
			saved_path = './final_model'
			
			# only restore graph inside a tensorflow session
			self.sess = tf.Session()

			# import the graph
			self.saver = tf.train.import_meta_graph(saved_path + '.meta', clear_devices=True)

			# restore the parameters
			self.saver.restore(self.sess, saved_path)

			self.graph = tf.get_default_graph()

			# retrieve variables from tensorflow graph
			self.X = self.graph.get_tensor_by_name("X:0")
			self.op_to_restore = self.graph.get_tensor_by_name("op_to_restore:0")
			self.dropout = self.graph.get_tensor_by_name("dropout:0")

	# inference using trained tensorflow graph, input data_in is 50 by 7, output traj_out is 1 by 7
	def tf_inference(self, data_in):
		data_in = np.asarray(data_in)
		data_in = np.reshape(data_in, [1, 50, 7])
		
		# traj_out is the filtered output of tensorflow graph
		traj_out = self.sess.run(self.op_to_restore, feed_dict={self.X: data_in, self.dropout: 1.0})
		traj_out = np.reshape(traj_out, [1, 7])
		
		return traj_out


def main():    
	RRN.UseNumPy = True
	
	# Create and Register Local Transport
	t1 = RR.LocalTransport()
	t1.StartServerAsNodeName("tf_rr")
	RRN.RegisterTransport(t1)
	
	# Create and Register TCP Transport
	t2 = RR.TcpTransport()
	t2.EnableNodeAnnounce()
	t2.StartServer(1234)
	RRN.RegisterTransport(t2)
	
	# read in Service Definition File
	with open('tf_rr.robdef','r') as f:
		service_def = f.read()
	
	# Register Service Definition
	RRN.RegisterServiceType(service_def)
	
	# Create instance of tensorflow object
	tf_obj = tfObject()
	
	# Register Service 'tfObject'
	RRN.RegisterService("tfObject","tf_rr.tfObject", tf_obj)
	   
	print "Connect to tfObject at:"
	# address : port / node name / service
	print "tcp://localhost:1234/tf_rr/tfObject"
	raw_input('press enter to quit')
	

if __name__ == '__main__':
	main()
