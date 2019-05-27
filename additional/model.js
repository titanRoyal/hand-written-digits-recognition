class Model {
  constructor(inn, h, o, lr = 0.06) {
    tf.tidy(() => {
      this.lr = lr;
      this.output = o;
      this.model = tf.sequential();
      for (var i = 0; i <= h.length; i++) {
        if (i == 0) {
          this.model.add(tf.layers.dense({
            inputShape: [inn],
            units: h[i],
            activation: "sigmoid"
          }));
        } else {
          this.model.add(tf.layers.dense({
            units: h[i - 1],
            activation: "sigmoid",
            bias: "true"
            //meansquareError
          }));
        }
      }
      this.model.add(tf.layers.dense({
        units: o,
        activation: "sigmoid",
        bias: "true"
      }));
      this.model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(this.lr)
      });
    })
  }
  predict(x, addtab = true) {
    if (addtab) {
      let end = [];
      tf.tidy(() => {
        let xx = tf.tensor2d([x]);
        let result = this.model.predict(xx).dataSync();
        result.map((data) => {
          end.push(data);
        })
      })
      return end;
    } else {
      let end = [];
      tf.tidy(() => {
        let xx = tf.tensor2d(x);
        let result = this.model.predict(xx).dataSync();
        result.map((data) => {
          end.push(data);
        })
        let minitab = [];
        let bigtab = [];
        for (var i = 0; i < end.length; i++) {
          minitab.push(end[i]);
          if ((i + 1) % this.output == 0) {
            bigtab.push(minitab);
            minitab = [];
          }
        }
        end = bigtab;
      })
      return end;
    }
  }
  async train(x, y, addtab = true) {
    let xx, yy
    if (addtab) {
      xx = tf.tensor2d([x]);
      yy = tf.tensor2d([y]);
    } else {
      xx = tf.tensor2d(x);
      yy = tf.tensor2d(y);
    }
    await this.model.fit(xx, yy).then(() => {epochs:100});
    tf.dispose(xx);
    tf.dispose(yy);
    return "done";
  }
}
