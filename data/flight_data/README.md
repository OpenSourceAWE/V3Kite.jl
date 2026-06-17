# Flight data

The flight data was moved to the repository https://github.com/OpenSourceAWE/FlightData

## Adding or removing flight logs 
- check out FlightData repo
- unpack the .tar.gz file
- change the content of the folder flight data
- create a new .tar.gz file from the folder
- commit and push
- create a new release and add the new file flight_data.tar.gz to the release
- update the Artifact.yaml file from the release (AI can do that)

## Creating a release

Most convenient on the command line:

```bash
cd repos/FlightData
gh release create  v0.1.1 flight_data.tar.gz --repo OpenSourceAWE/FlightData
```
Adapt the version number in the command above!

The `Artifact.yaml` file that needs to be updated lives at three locations:
- in the FlightData repo for documentation purposes
- in README.md of the FlightData repo for documentation purposes
- in the examples folder of this repo. This is the only location where it is actually used.