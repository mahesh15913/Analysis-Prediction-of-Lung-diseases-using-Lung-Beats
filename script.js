// // Set up variables
// var audioContext;
// var recorder;
// var audioStream;
// var recording = false;

// // Set up event listeners
// document.getElementById('recordButton').addEventListener('click', function() {
//   if (recording) {
//     stopRecording();
//   } else {
//     startRecording();
//   }
// });

// document.getElementById('stopButton').addEventListener('click', stopRecording);

// // Set up audio context and recorder
// function startRecording() {
//   if (!audioContext) {
//     audioContext = new (window.AudioContext || window.webkitAudioContext)();
//   }

//   if (!recorder) {
//     recorder = new Recorder(audioContext.createMediaStreamSource(audioStream), {
//       numChannels: 1
//     });
//   }

//   if (!recording) {
//     recording = true;
//     recorder.record();
//     document.getElementById('recordButton').innerHTML = 'Stop';
//     document.getElementById('stopButton').disabled = false;
//   }
// }

// function stopRecording() {
//   if (recording) {
//     recording = false;
//     recorder.stop();
//     recorder.exportWAV(function(blob) {
//       var url = URL.createObjectURL(blob);
//       var li = document.createElement('li');
//       var au = document.createElement('audio');
//       au.controls = true;
//       au.src = url;
//       li.appendChild(au);
//       document.getElementById('recordingsList').appendChild(li);
//     });
//     document.getElementById('recordButton').innerHTML = 'Record';
//     document.getElementById('stopButton').disabled = true;
//   }
// }

// // Request audio input
// navigator.mediaDevices.getUserMedia({ audio: true })
//  .then(function(stream) {
//     audioStream = stream;
//   })
//  .catch(function(err) {
//     console.log('Error getting audio stream: ' + err);
//   });

// Set up variables
var audioInput = document.getElementById('audioInput');
var recordButton = document.getElementById('recordButton');
var stopButton = document.getElementById('stopButton');
var recordingsList = document.getElementById('recordingsList');
var recording = false;
var audioContext;
var mediaRecorder;
var audioBlob;

// Set up event listeners
recordButton.addEventListener('click', function() {
  if (recording) {
    stopRecording();
  } else {
    startRecording();
  }
});

stopButton.addEventListener('click', stopRecording);

audioInput.addEventListener('change', function(event) {
  var file = event.target.files[0];
  var reader = new FileReader();

  reader.onload = function(event) {
    var arrayBuffer = event.target.result;
    audioContext.decodeAudioData(arrayBuffer, function(audioBuffer) {
      var audioData = audioBuffer.getChannelData(0);
      var wavBlob = new Blob([toWav(audioData)], { type: 'audio/wav' });
      audioBlob = wavBlob;
      recordButton.disabled = false;
      stopButton.disabled = true;
    });
  };

  reader.readAsArrayBuffer(file);
});

// Set up audio context and recorder
function startRecording() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  if (!mediaRecorder) {
    mediaRecorder = new MediaRecorder(audioContext.createMediaStreamSource(audioInput.captureStream()));
    mediaRecorder.ondataavailable = function(event) {
      if (event.data.size > 0) {
        var reader = new FileReader();

        reader.onload = function(event) {
          var arrayBuffer = event.target.result;
          audioContext.decodeAudioData(arrayBuffer, function(audioBuffer) {
            var audioData = audioBuffer.getChannelData(0);
            var wavBlob = new Blob([toWav(audioData)], { type: 'audio/wav' });
            audioBlob = wavBlob;
            recordButton.disabled = false;
            stopButton.disabled = true;
          });
        };

        reader.readAsArrayBuffer(event.data);
      }
    };
  }

  if (!recording) {
    recording = true;
    mediaRecorder.start();
    recordButton.innerHTML = 'Stop';
    stopButton.disabled = false;
  }
}

function stopRecording() {
  if (recording) {
    recording = false;
    mediaRecorder.stop();
    mediaRecorder.stream.getAudioTracks()[0].stop();
    recordButton.innerHTML = 'Record';
    stopButton.disabled = true;
  }
}

// Convert audio data to WAV format
function toWav(audioData) {
  var wavFile = new Uint8Array(audioData.length + 44);
  var view = new DataView(wavFile.buffer);

  // RIFF identifier
  view.setUint32(0, 1380533830, false);
  // RIFF chunk length
  view.setUint32(4, 36 + audioData.length, true);
  // WAVE identifier
  view.setUint32(8, 1463899717, false);
  // fmt subchunk identifier
  view.setUint32(12, 1718449184, false);
  // fmt subchunk length
  view.setUint32(16, 16, true);
  // audio format (PCM)
  view.setUint16(20, 1, true);
  // number of channels
  view.setUint16(22, 1, true);
  // sample rate
  view.setUint32(24, audioContext.sampleRate, true);
  // byte rate
  view.setUint32(28, audioContext.sampleRate * 2, true);
  // block align
  view.setUint16(32, 2, true);
  // bits per sample
  view.setUint16(34, 16, true);
  // data subchunk identifier
  view.setUint32(36, 1684108385, false);
  // data subchunk length
  view.setUint32(40, audioData.length, true);

  for (var i = 0; i < audioData.length; i++) {
    view.setUint8(44 + i,