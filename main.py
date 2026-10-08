import time
from argparse import ArgumentParser

STEPS = ['sat', 'model', 'scatter', 'map', 'timeseries', 'tracks']

parser = ArgumentParser(description='WW3 validation')
parser.add_argument('-c', '--configuration', default='conf.yaml', help='catalog file')
parser.add_argument('-s', '--start_date', required=True, help='start date YYYYMMDD')
parser.add_argument('-e', '--end_date', required=True, help='end date YYYYMMDD')
parser.add_argument('--steps', nargs='+', choices=STEPS, default=STEPS,
                    help='steps to run (default: all)')
parser.add_argument('-w', '--workers', type=int, default=None,
                    help='processes for reading satellite files (default: LSF allocation or 1)')
args = parser.parse_args()
conf_path, start_date, end_date = args.configuration, args.start_date, args.end_date


def run(step):
    t0 = time.time()
    if step == 'sat':
        from satellite_preprocessing import Sat_processer
        Sat_processer(conf_path, start_date, end_date, workers=args.workers).run()
    elif step == 'model':
        from model_preprocessing import submit
        submit(conf_path, start_date, end_date)
    elif step == 'scatter':
        from validation import main
        main(conf_path, start_date, end_date)
    elif step == 'map':
        from validation_map import main
        main(conf_path, start_date, end_date)
    elif step == 'timeseries':
        from validation_timeseries import main
        main(conf_path, start_date, end_date)
    elif step == 'tracks':
        from validation_tracks import main
        main(conf_path, start_date, end_date)
    print(f'=== {step} done in {time.time() - t0:.1f} s ===')


for step in STEPS:
    if step in args.steps:
        run(step)
