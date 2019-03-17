"""
Progress bar for folks who mean buzinaz
"""
from datetime import datetime

class Progress():
  """ Pretty print progress for neural net training """
  def __init__(self, batches, best_val=0, test_val=0, epoch=-1,
                            progress_bar=True, bar_length=30, track_best=True):
    self.progress_bar = progress_bar # boolean
    self.bar_length = bar_length
    self.t1 = datetime.now()
    self.train_start_time = self.t1
    self.batches = batches
    self.current_batch = 0
    self.last_eval = '' # save last eval to add after train
    self.last_train = ''
    self.track_best = track_best
    self.best_val = best_val
    self.test_val = test_val
    self.epoch = epoch
    self.pause_bar = False
    self.current_episode = 0
    self.t1 = datetime.now()

  def epoch_start(self):
    print()
    self.epoch += 1
    self.current_batch = 0 # reset batch

  def train_end(self):
    print()

  def print_train(self, episode, total_timesteps, reward, running_avg_rw, temp):
    if self.current_episode is not episode:
      self.current_episode = episode
      self.t1=datetime.now()
    t2 = datetime.now()
    episode_time = (t2 - self.t1).total_seconds()
    total_time = (t2 - self.train_start_time).total_seconds()/60
    self.last_train='Episode {:>5.0f}: | steps {:>8.0f} | tot min: {:>5.1f} | reward: {:>7.2f} av reward: {:>6.2f} | temp: {:>1.4f}'.format(
        episode, total_timesteps, total_time, reward, running_avg_rw, temp)
    # self.last_train='{:2.0f}: sec: {:>5.0f} | total min: {:>5.1f} | disc: {:>3.4f} gen: {:>3.4f} | ent: {:>3.0f}'.format(
        # self.epoch, epoch_time, total_time, d_loss, g_loss, ent)
    print(self.last_train, end='')
    self.print_bar(total_timesteps)
    print(self.last_eval, end='\r')

  def print_cust(self, msg):
    """ Print anything, append previous """
    print(msg, end='')

  def test_best_val(self, te_acc):
    """ Test set result at the best validation checkpoint """
    self.test_val = te_acc

  def print_end_epoch(self, msg):
    """ Print after training , then new line """
    print(self.last_train, end='')
    print(msg, end='\r')

  def print_eval(self, value):
    # Print last training info
    print(self.last_train, end='')
    self.last_eval = '| last val: {:>3.4f} '.format(value)

    # If tracking eval, update best
    extra = ''
    if self.track_best == True:
      if value > self.best_val:
        self.best_val = value
      self.last_eval += '| best val: {:>3.4f} | test on best model: {:>3.4f}'.format(self.best_val, self.test_val)
    print(self.last_eval, end='\r')

  def print_bar(self, total_timesteps):
    if self.pause_bar == False:
      self.current_batch = total_timesteps - self.epoch*self.batches
    bars_full = int(self.current_batch/self.batches*self.bar_length)
    bars_empty = self.bar_length - bars_full
    progress ="| [{}{}] ".format(u"\u2586"*bars_full, '-'*bars_empty)
    self.last_train += progress
    print(progress, end='')
