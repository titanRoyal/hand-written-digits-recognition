let training_img = [],
  training_label = [],
  testing_img = [],
  testing_label = []
const size = 784
let model = new neural_net(size, [60], 10, 0.001)
let index = 0

function preload() {
  loadBytes("meta/data/t-img.byte", (data) => {
    let offset = 16
    for (var i = 0; i < 10000; i++) {
      testing_img[i] = []
      for (var i1 = 0; i1 < size; i1++) {
        testing_img[i].push(data.bytes[size * i + i1 + offset])
      }
    }
  })
  loadBytes("meta/data/tr-img.byte", (data) => {
    let offset = 16
    for (var i = 0; i < 60000; i++) {
      training_img[i] = []
      for (var i1 = 0; i1 < size; i1++) {
        training_img[i].push(data.bytes[size * i + i1 + offset])
      }
    }
  })
  loadBytes("meta/data/t-lab.byte", (data) => {
    let offset = 8
    for (var i = 0; i < 10000; i++) {
      let tab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      tab[data.bytes[offset + i]] = 1
      testing_label.push(tab)
    }
  })
  loadBytes("meta/data/tr-lab.byte", (data) => {
    let offset = 8
    for (var i = 0; i < 60000; i++) {
      let tab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      tab[data.bytes[offset + i]] = 1
      training_label.push(tab)
    }
  })
}
//let img

function setup() {
  createCanvas(400, 400);
  // img = createImage(28, 28)
  // img.loadPixels()
  // for (var i = 0; i < size; i++) {
  //   img.pixels[i * 4 + 0] = 255
  //   img.pixels[i * 4 + 1] = 255
  //   img.pixels[i * 4 + 2] = 255
  //   img.pixels[i * 4 + 3] = training_img[10][i]
  // }
  // img.updatePixels()
  background(0);
  stroke(255)
  strokeWeight(15)
}

function mouseDragged() {
  line(pmouseX, pmouseY, mouseX, mouseY);
}

function checkCanvas() {
  let img = get()
  img.resize(28, 28)
  img.loadPixels()
  let tab = []
  img.pixels.forEach((data, index) => {
    if (index % 4 == 0) {
      tab.push(data)
    }
  })
  let pred = model.feedforward(tab)
  console.log(pred.indexOf(max(pred)));
}

function custumTrain() {
  let tab = [],
    label = []
  while (tab.length < 4000) {
    let f = floor(random(60000));
    let pred = model.feedforward(training_img[f])
    if (pred.indexOf(max(pred)) != training_label[f].indexOf(1)) {
      tab.push(training_img[f])
      label.push(training_label[f])
    }
  }
  tab.forEach((data, index) => {

    model.train(data, label[index]);

  })
}

function train() {
  let tab = [],
    label = []
  while (tab.length < 4000) {
    let f = floor(random(60000));
    if (!tab.includes(training_img[f])) {

      tab.push(training_img[f])
      label.push(training_label[f])
    }
  }
  tab.forEach((data, index) => {

    model.train(data, label[index]);

  })
}

function test() {
  let pred, win = 0,
    lose = 0
  testing_img.forEach((data, index) => {
    pred = model.feedforward(data);
    if (pred.indexOf(max(pred)) == testing_label[index].indexOf(1)) {
      win++;
    } else {
      lose++;
    }
  })
  return {
    win,
    lose
  }
}
async function all_in(x = 1, typi = "n") {
  if (typi == "n") {
    for (var i = 0; i < x; i++) {
      await train();
      console.log("done");
    }
  } else {
    for (var i = 0; i < x; i++) {
      await custumTrain();
      console.log("done epoch");
    }
  }

}
