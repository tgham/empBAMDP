
# DayanTurk Lite

## Motivation

This is the current state of the "default" backend for running online
experiments in AGPD.

The normal setup we use is to have a strict separation of frontend and backend
process. The frontend is usually a collection of html, javascript, and css files,
built using a toolkit like `jspsych`, which will be served by `apache` on a
particular address of VM (something like
`https://kyblab2.tuebingen.mpg.de/fancy_experiment`). 

The backend (this project) is a `fastapi`-based python application (which is run
on the same VM, listening on a different port), and which listens for
connections on specific routes at that port (*e.g.*,
`https://kyblab2.tuebingen.mpg.de:9000/create`) and runs a few simple functions.

## API details

The backend comes with the following routes:

- `/`: This is a status route, which can be connected to from a browser to check
  the backend is running

- `/create/{pid}`:
    - The route for making the backend aware of a new participant
    - It takes the **Prolific ID** of the participant
    - It returns a randomised ID which will represent that participant in our
    experiment

- `/complete/{ppt_id}`:
    - This route allows the saving of the final data for a participant
    - It takes a participant ID (as generated above), and a json payload of that
      participants data
    - It saves the data to the `incomplete` directory, and then moves it to the
      `complete` directory

- `/incomplete/{ppt_id}`:
    - This route allows the saving of in-progress data for a participant
    - It takes a participant ID (as generated above), and a json payload of that
      participants data up to this point
    - It saves the data to the `incomplete` directory

- `/complete_success/{ppt_id}` and `/complete_failure/{ppt_id}`:
    - These routes finalise a participant without changing the data that has
      already been received
    - They move the `incomplete` data (if any) to either the `complete` dictory
      (in case of success) or the `invalid` directory (in case of failure)


`dev/backend_integration.js` contains the javascript functions that the frontend
can use to send messages to these routes. Specifically, it defines the following
functions:

- `create_participant(pid)`: this takes the prolific id, asks the backend to
create a new participant, and returns the newly created randomised id

- `send_complete(id, data)` and `send_incomplete(id, data)`:
    - These functions take the id (as generated above), and all of the data up
      to this point for that participant, and send that data to the `complete` or
      `incomplete` endpoints respectively

- `complete(id, success)`: this function hits either the `complete_success` or
  `complete_failure` end point, depending on the value of success

There is also `get_prolific_id()` which is a utility function for reading the
prolific id from the URL when a participant connects. 

## Workflow

The communication with the backend is designed to be simple. The
`dev/backend_integration.js` should be copied into the frontend project, and
sourced in the `index.html` file (or whatever your main file is called).

You probably have a `script` block in your main html file, and you can initiate
contact with the backend by including something like:

```js
var pid = get_prolific_id();
let id = null;

//Tell the backend about the new ppt
create_participant(pid).then((value) => {
  // If the id is null, then the backend is complaining that the ppt has
  // done the experiment before
  // You should probably bail out in that case
  if (value['id'] == null) {
    console.log(`${pid} is not unique`);
  }
  // Otherwise, give the ppt an internal id
  id = value['id'];
  console.log(`id => ${id}`);
})
```

This will give you an `id` which you can use to refer to that participant from
then on. An example `jspsych` setup for using the backend might look like the
following:

```js
var jsPsych = initJsPsych({
  on_trial_finish: function() {
    // Get the participant's data so far
    var ppt_data = jsPsych.data.get().json();

    send_incomplete(id, ppt_data);

  },
  on_finish: function () {
    // Get all of the participant's data
    var ppt_data = jsPsych.data.get().json();

    send_complete(id, ppt_data);
    setTimeout(function () {
      // Redirect to another website
      window.location.replace(SOME_OTHER_WEBSITE);
    }, 1000);
  }
});
```

## Configuration

There is a minimal amount of configuration you can use, which is defined in
`app.json`. It is mostly restricted to naming files and directories. Other
configuration can be added as needed.

By default, data is saved in a `data` subdirectory, with different
sub-subdirectories for `complete`, `incomplete`, and `invalid` data. There is
also a `metadata` subdirectory which, by default, contains a file which maps
from prolific ids to randomised ids.


## Usage (Testing)

The backend is designed to be able to test locally (using `http`). The process
is as follows:

- Clone the repository somewhere
- Create a python virtual environment and activate it
  - *e.g.*, `python -m venv .env` and `source .env/bin/activate`
- Install the necessary libraries
  - *e.g.*, `pip install -r requirements.txt`
- Run the backend with `python main.py --local`
- Make a note of the IP address is it using
  - *e.g.*, `http://127.0.0.1:8000`

If you open `localhost:8000` in a web browser, you should see a message saying
that the backend is running.

The frontend can also be run locally. Assuming that you have added the
`backend_integration.js` to your experiment, and included the javascript code
shown above, you should be able to do the following:

- Change into the directory containing your main html file
- Run `python -m http.server 3000` to start serving the frontend
- Opening `localhost:3000?PROLIFIC_PID=test` should then run your experiment
locally, including sending data to the backend for saving.

## Usage

The actual usage of the backend is similar to the testing usage, but differs in
the following ways:

- It is run with `python main.py`
- It uses `https` for communication
- It runs on a VM as mentioned above
- It requires that `SSL_CERTFILE` and `SSL_KEYFILE` environment variables are
set beforehand, which point to valid certificate and private key file (as
generated by a certificate authority):

```sh
export SSL_CERTFILE=/etc/letsencrypt/.../cert.pem
export SSL_KEYFILE=/etc/letsencrypt/.../privkey.pem
python main.py
```

These steps might seem a bit confusing, but, at this point you should be in
contact with Andrew, and he will help you work through them.
