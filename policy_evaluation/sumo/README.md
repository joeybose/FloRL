# Sumo Traffic Simulator
SUMO is an open source traffic simulator. Details can be found in the appendix of the paper

## Dependency
* [sumo](http://www.sumo.dlr.de/userdoc/Installing.html) version at least 0.32.0
* tensorflow 1.0 or above

## Example
```sh
$ python collect_date.py 0 3000
$ python evaluate.py --nt 250 --ts 400 --bp 2
```
Here 'nt' denotes number of iteration, 'ts' denotes truncated size, 'bp' denoted behavior policy ID