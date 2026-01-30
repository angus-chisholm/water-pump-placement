# Water fountain, pipe and pump placement optimisation
> Modelling the addition of new water fountains to an existing rural water system to include piped networks with pumps. Cost is evaluated against impact: 1. person-weighted distance travelled to get water at the nearest source or 2. percentage of people within 30 mins round trip of a safely managed source 'basic quality' (UN, UNICEF)


## Installation

Download and run:
```sh
pip install -r requirements.txt
```
And clone this repository.

## Usage example
Make sure to change the file path in lines 1642 (dataset) and 639 (consumption/usage profile generator)

Run the code `opti_multi_pump_clean.py` in your terminal of choice. The working directory should be the water-pump-placement folder.

After selection of the optimisation parameters, the code will run, and with ~100 generations and 3 fountains it takes approximately 5-10 minutes to complete. Once complete, the files should save in the 'Output' folder, and the interactive plots (3 plots) should appear.

To re-view the plots, use the `output_plotter.ipynb` file.
