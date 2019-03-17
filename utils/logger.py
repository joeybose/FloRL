from comet_ml import Experiment
import os
import datetime
import sys
import pprint
import time
import json
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, args, path='logs/', name=''):
        print("logging experiment " + name)
        self.log_tb = args.log_tb
        self.log_comet = args.comet

        if args.comet:
            experiment = Experiment(api_key=args.comet_apikey,\
                project_name="learned-behaviour-policy",auto_output_logging="None",\
                workspace=args.comet_username,auto_metric_logging=False,\
                auto_param_logging=False)
            experiment.set_name(args.namestr)
            args.experiment = experiment

        # Setup Comet

        # if self.log_comet:
        #     if not os.path.isfile("settings.json"):
        #         sys.exit("You need a 'settings.json' file to use Comet")
        #     with open('settings.json') as f:
        #         data = json.load(f)
        #         api_key = data["comet"]["api_key"]
        #         project_name = data["comet"]["project_name"]
        #         workspace = data["comet"]["workspace"]

        #     self.comet = Experiment(
        #         api_key=api_key,
        #         project_name=project_name,
        #         workspace=workspace)
        #     self.comet.set_name(args.namestr)

        # Setup tensorboad
        if args.log_tb:
            # Prep for local logging
            if not os.path.isdir(path):
                os.mkdir(path)
            log_file = os.path.join(path, name)
            # Check if exists
            if os.path.isdir(log_file):
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime(
                    '%Y-%m-%d_%H:%M:%S')
                name += "_" + st
                log_file = os.path.join(path, name)
            self.tblogger = SummaryWriter(log_dir=log_file)
            self.log_text('logfile location', str(log_file))

        # COMET has "notes/markdown" option on website, but can't figure out which call to use
        # Initial log information
        self.log_text('comment', args.comment)
        self.log_text('run command', " ".join(sys.argv))
        flags = pprint.pformat(vars(args))
        self.log_text('Flags', flags)
        self.log_text('chkpt', 'chkpt/model_path')
        self.log_time('start time', 0)

    def log_text(self, tag, msg):
        if self.log_comet:
            experiment.log_html("<p>{}: {}</p>".format(tag, msg))
        if self.log_tb:
            self.tblogger.add_text(tag, msg, global_step=0)

    def log_time(self, msg, step):
        """
        Log time only for tensorboard, Comet does it auto
        """
        if self.log_tb:
            cur = datetime.datetime.now()
            msg = msg + " " + str(cur)
            self.tblogger.add_text('time', msg, step)

    def log(self, tag, scalar, step):
        if self.log_comet:
            experiment.log_metric(tag, scalar, step=step)
        if self.log_tb:
            self.tblogger.add_scalar(tag, scalar, step)

    def log_image(self, tag, image, epoch):
        """
        Save image to tensorboard. Tb expects image with dimensions: CHW, so
        must reshuffle current format of HWC
        """
        if self.log_tb:
            image = image.transpose(2, 0, 1)
            self.tblogger.add_image(tag, image, epoch)
        if self.log_comet:
            #TODO: not sure about comet. Looks like need to save file first
            # then call experiment.log_image(file_path)
            pass

    def log_histogram(self, tag, param, epoch):
        if self.log_tb:
            self.tblogger.add_histogram(tag, param, epoch)
