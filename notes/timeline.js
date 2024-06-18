function setup() {
  createCanvas(900, 200);
  textAlign(CENTER, CENTER);
  textSize(18);
  pixelDensity(1);
  intervalWidth = 100
  
  // Create a button for saving the canvas 
  removeBtn = createButton("Save Canvas"); 
  removeBtn.position(200, 400) 
  removeBtn.mousePressed(saveToFile); 

  
  let interArrivalTimes = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];  // Inter-arrival times
  let serviceTimes = [0.6, 0.7, 0.5, 0.4, 0.0, 0.9, 0.7, 1.2];   // Service times for each process
  let waitingTimes = [0.0];  // Vector with waiting times
  let waitingIntervals = [[0, 0]]
  let idleTimes = [];  // Vector with waiting times
  let idleIntervals = []
  let t = 0.0;  // Planned schedule progression
  let rt = 0.0;  // Real schedule progression
  let totalTime = interArrivalTimes.reduce((accumulator, currentValue) => accumulator + currentValue, 0) + 1;
  let startTime = 150
  let end = startTime + totalTime * intervalWidth 
  
  // Calculate waiting times
  for (let i = 1; i < interArrivalTimes.length; i++) {
    rt += serviceTimes[i - 1];
    rt = round(rt, 1) // Javascript has a floatation error
    t += interArrivalTimes[i];
    waitingTimes.push(round(Math.max(0, rt - t), 1));
    waitingIntervals.push([t, t + waitingTimes[i]]);
    idleTimes.push(round(Math.max(0.0, t - rt), 1))
    rt += round(Math.max(0.0, t - rt), 1)
    print(`i = ${i}, t = ${t}, rt = ${rt}, waitingTimes[i] = ${waitingTimes[i]}, waitingInterval= ${waitingIntervals[i]}, serviceTimes[i] = ${serviceTimes[i]}`);
  }

  print("Final waiting times:", waitingTimes);
  print("Final idle times:", idleTimes);
  
  // Draw timeline
  stroke(0);
  line(startTime, 150, end, 150);
  
  for (let i = 0; i <= totalTime; i++) {
    let x = map(i, 0, totalTime, startTime, end);
    line(x, 145, x, 155);
    if(i == totalTime) {
      text('T = ' + i, x, 170);
    } else {
      text('' + i, x, 170);
    }
    
  }
  
  // Draw waiting times
  for (let i = 0; i < waitingTimes.length; i++) {
    if (waitingTimes[i] > 0) {
      noStroke()
      fill(150, 128);
      rect(startTime + waitingIntervals[i][0] * intervalWidth, 50, waitingTimes[i] * intervalWidth, 25);
    }
  }
  
  // Draw overtime
  overTime = round(Math.max(0, rt + serviceTimes[serviceTimes.length - 1] - totalTime), 1)
  print(overTime)
  rect(startTime + totalTime * intervalWidth, 50, overTime * intervalWidth, 25);
  
  // Draw service times
  x = startTime;
  for (let i = 0; i < serviceTimes.length; i++) {
    stroke(255);  // Set stroke color to white
    fill(0);
    rect(x, 75, serviceTimes[i] * intervalWidth, 25);
    fill(255);
    noStroke(); // Disable stroke for text
    if(serviceTimes[i] > 0.0) {
      if(i + 1 == serviceTimes.length) {text('N = ' + (i + 1), x + serviceTimes[i] * intervalWidth / 2, 87.5)} else {text(i + 1, x + serviceTimes[i] * intervalWidth / 2, 87.5);}
      
    }
    stroke(255);  // Re-enable stroke for the next rectangle
    x += serviceTimes[i] * intervalWidth;
    
    if (i < idleTimes.length && idleTimes[i] > 0) {
      x += idleTimes[i] * intervalWidth;
    }
  }
  
  // Draw idle times
  x = startTime;
  stroke(0);
  for (let i = 0; i < idleTimes.length; i++) {
    if (idleTimes[i] > 0) {
      fill(255);
      rect(x + serviceTimes[i] * intervalWidth, 100, idleTimes[i] * intervalWidth, 25);
      x += (serviceTimes[i] + idleTimes[i]) * intervalWidth;
    } else {
      x += serviceTimes[i] * intervalWidth;
    }
  }
  
  // Draw labels
  fill(0);
  noStroke();
  text('Waiting times', 60, 62.5);
  text('Service times', 60, 87.5);
  text('Idle times', 75, 112.5);

}

function saveToFile() { 
  // Save the current canvas to file as png 
  saveCanvas('timeline.png') 
}

function draw() {
  // Nothing needed here
}
