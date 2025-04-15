import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.misc.visualizer import plot_training_metrics 

plot_training_metrics('output/ouputlogs_topk200.txt')