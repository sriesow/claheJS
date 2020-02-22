export class claheEqualization {
  constructor(limit, tilesX, tilesY) {
    this.tilesX = tilesX;
    this.tilesY = tilesY;
    this.limit = limit;
    this.histSize = 256;
    this.tileSize = [];
  }

  async convertCLAHE(imageData, width, height) {
    let hueArray = [];
    let saturationArray = [];
    let luminanceArray = [];
    for (let i = 0; i < imageData.length; i = i + 4) {
      let HSLObject = this.RGBToHSL(
        imageData[i],
        imageData[i + 1],
        imageData[i + 2]
      );
      hueArray.push(HSLObject[0]);
      saturationArray.push(HSLObject[1]);
      luminanceArray.push(HSLObject[2]);
    }

    let myCanvas = document.createElement('canvas');
    myCanvas.width = width;
    myCanvas.height = height;
    let newImageData = myCanvas
      .getContext('2d')
      .createImageData(myCanvas.width, myCanvas.height);

    inputArray = luminanceArray;
    let lutArray = this.calculateLUTBody(inputArray, height, width);
    let outputArray = this.calculateInterpolationBody(
      inputArray,
      lutArray,
      height,
      width
    );
    let claheModifiedArray = outputArray.flatten().tolist();

    for (let y = 0; y < myCanvas.height; y++) {
      for (let x = 0; x < myCanvas.width; x++) {
        let a = (y * myCanvas.width + x) * 4;
        newImageData.data[a] = hueArray[a];
        newImageData.data[a + 1] = saturationArray[a];
        newImageData.data[a + 2] = claheModifiedArray[a];
        newImageData.data[a + 3] = 1;
      }
    }
    let RGBFinalValues = this.hslToImageData(newImageData.data, width, height);
    return RGBFinalValues;
  }

  RGBToHSL(r, g, b) {
    (r /= 255), (g /= 255), (b /= 255);
    var max = Math.max(r, g, b),
      min = Math.min(r, g, b);
    var h,
      s,
      l = (max + min) / 2;

    if (max == min) {
      h = s = 0; // achromatic
    } else {
      var d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case r:
          h = (g - b) / d + (g < b ? 6 : 0);
          break;
        case g:
          h = (b - r) / d + 2;
          break;
        case b:
          h = (r - g) / d + 4;
          break;
      }
      h /= 6;
    }

    return [parseInt(h * 360), parseInt(s * 100), parseInt(l * 100)];
  }

  hslToRgb(h, s, l) {
    var r, g, b;
    var h = h / 360;
    var s = s / 100;
    var l = l / 100;
    if (s == 0) {
      r = g = b = l; // achromatic
    } else {
      var hue2rgb = function hue2rgb(p, q, t) {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };

      var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      var p = 2 * l - q;
      r = hue2rgb(p, q, h + 1 / 3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1 / 3);
    }

    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }

  hslToImageData(pixels, width, height) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);

    // Iterate through every pixel
    for (let i = 0; i < pixels.length; i += 4) {
      // Modify pixel data
      let RGB = this.hslToRgb(pixels[i], pixels[i + 1], pixels[i + 2]);
      imageData.data[i + 0] = RGB[0]; // R value
      imageData.data[i + 1] = RGB[1]; // G value
      imageData.data[i + 2] = RGB[2]; // B value
      imageData.data[i + 3] = 255;
    }
    return imageData;
  }

  calculateLUTBody(inputArray, height, width) {
    this.tileSize[0] = width / this.tilesX;
    this.tileSize[1] = height / this.tilesY;

    let srcForLut = inputArray.clone();
    let tileHist = nj.zeros([this.tilesX * this.tilesY, this.histSize]);
    let lutArray = nj.zeros([this.tilesX * this.tilesY, this.histSize]);

    let tileSizeTotal = this.tileSize[0] * this.tileSize[1];
    const lutScale = (this.histSize - 1) / tileSizeTotal;

    let clipLimit = 0;
    if (this.limit > 0.0) {
      clipLimit = parseInt((this.limit * tileSizeTotal) / this.histSize);
      clipLimit = Math.max(clipLimit, 1);
    }

    // Calculate Histogram
    for (let i = 0; i < this.tilesY; i++) {
      for (let j = 0; j < this.tilesX; j++) {
        let startX = i * this.tileSize[0];
        let endX = startX + this.tileSize[0];
        let startY = j * this.tileSize[1];
        let endY = startY + this.tileSize[1];
        let numBlock = i + this.tilesX * j;
        for (let ii = startX; ii < endX; ii++) {
          for (let jj = startY; jj < endY; jj++) {
            let index = srcForLut.get(jj, ii);
            let tileHistVal = tileHist.get(numBlock, index);
            tileHist.set(numBlock, index, tileHistVal + 1);
          }
        }

        // Clip Histogram

        if (clipLimit > 0) {
          // How many pixels were clipped
          let clipped = 0;
          for (let i = 0; i < this.histSize; i++) {
            let histValue = tileHist.get(numBlock, i);
            if (histValue > clipLimit) {
              clipped += histValue - clipLimit;
              tileHist.set(numBlock, i, clipLimit);
            }
          }
          // Redistribute clipped pixels
          let redistBatch = clipped / this.histSize;
          let residual = clipped - redistBatch * this.histSize;
          for (let i = 0; i < this.histSize; i++) {
            let histValue = tileHist.get(numBlock, i);
            tileHist.set(numBlock, i, histValue + redistBatch);
          }
          if (residual != 0) {
            let residualStep = Math.max(this.histSize / residual, 1);
            let i = 0;
            while (i < this.histSize && residual > 0) {
              let histValue = tileHist.get(numBlock, i);
              tileHist.set(numBlock, i, histValue + 1);
              i += residualStep;
              residual--;
            }
          }
        }

        // Calculate lut
        lutArray.set(numBlock, i, tileHist.get(numBlock, 0) / tileSizeTotal);
        for (let i = 1; i < this.histSize; i++) {
          let tempVal =
            lutArray.get(numBlock, i - 1) +
            tileHist.get(numBlock, i) / tileSizeTotal;
          lutArray.set(numBlock, i, tempVal);
        }
      }
    }
    return lutArray;
  }

  calculateInterpolationBody(srcArray, lutArray, height, width) {
    let dstArray = srcArray.clone();
    for (let i = 0; i < width; i++) {
      for (let j = 0; j < height; j++) {
        if (i <= this.tileSize[0] / 2 && j <= this.tileSize[1] / 2) {
          let tempVal = 0;
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(lutArray.get(tempVal, srcVal) * 255);
          dstArray.set(j, i, finalVal);
        } else if (
          i <= this.tileSize[0] / 2 &&
          j >= (this.tilesX - 1) * this.tileSize[1] + this.tileSize[1] / 2
        ) {
          let tempVal = this.tilesX * (this.tilesX - 1);
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(lutArray.get(tempVal, srcVal) * 255);
          dstArray.set(j, i, finalVal);
        } else if (
          i >= (this.tilesX - 1) * this.tileSize[0] + this.tileSize[0] / 2 &&
          j <= this.tileSize[1] / 2
        ) {
          let tempVal = this.tilesX - 1;
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(lutArray.get(tempVal, srcVal) * 255);
          dstArray.set(j, i, finalVal);
        } else if (
          i >= (this.tilesX - 1) * this.tileSize[0] + this.tileSize / 2 &&
          j >= (this.tilesX - 1) * this.tileSize[1] + this.tileSize[1] / 2
        ) {
          let tempVal = this.tilesX * this.tilesX - 1;
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(lutArray.get(tempVal, srcVal) * 255);
          dstArray.set(j, i, finalVal);
        } else if (i <= this.tileSize[0] / 2) {
          let tempi = 0;
          let tempj = parseInt((j - this.tileSize[1] / 2) / this.tileSize[1]);
          let temp1 = parseInt(tempj * this.tilesX + tempi);
          let temp2 = parseInt(temp1 + this.tilesX);
          let p =
            (j - (tempj * this.tileSize[1] + this.tileSize[1] / 2)) /
            (1.0 * this.tileSize[1]);
          let q = 1 - p;
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(
            (q * lutArray.get(temp1, srcVal) +
              p * lutArray.get(temp2, srcVal)) *
              255
          );
          dstArray.set(j, i, finalVal);
        } else if (
          i >=
          (this.tilesX - 1) * this.tileSize[0] + this.tileSize[0] / 2
        ) {
          let tempi = this.tilesX - 1;
          let tempj = parseInt((j - this.tileSize[1] / 2) / this.tileSize[1]);
          let temp1 = parseInt(tempj * this.tilesX + tempi);
          let temp2 = parseInt(temp1 + this.tilesX);
          let p =
            (j - (tempj * this.tileSize[1] + this.tileSize[1] / 2)) /
            (1.0 * this.tileSize[1]);
          let q = 1 - p;
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(
            (q * lutArray.get(temp1, srcVal) +
              p * lutArray.get(temp2, srcVal)) *
              255
          );
          dstArray.set(j, i, finalVal);
        } else if (j <= this.tileSize[1] / 2) {
          let tempi = parseInt((i - this.tileSize[0] / 2) / this.tileSize[0]);
          let tempj = 0;
          let temp1 = parseInt(tempj * this.tilesX + tempi);
          let temp2 = parseInt(temp1 + 1);
          let p =
            (i - (tempi * this.tileSize[0] + this.tileSize[0] / 2)) /
            (1.0 * this.tileSize[0]);
          let q = 1 - p;
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(
            (q * lutArray.get(temp1, srcVal) +
              p * lutArray.get(temp2, srcVal)) *
              255
          );
          dstArray.set(j, i, finalVal);
        } else if (
          j >=
          (this.tilesX - 1) * this.tileSize[1] + this.tileSize[1] / 2
        ) {
          let tempi = parseInt((i - this.tileSize[0] / 2) / this.tileSize[0]);
          let tempj = this.tilesX - 1;
          let temp1 = parseInt(tempj * this.tilesX + tempi);
          let temp2 = parseInt(temp1 + 1);
          let p =
            (i - (tempi * this.tileSize[0] + this.tileSize[0] / 2)) /
            (1.0 * this.tileSize[0]);
          let q = 1 - p;
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(
            (q * lutArray.get(temp1, srcVal) +
              p * lutArray.get(temp2, srcVal)) *
              255
          );
          dstArray.set(j, i, finalVal);
        } else {
          let tempi = parseInt((i - this.tileSize[0] / 2) / this.tileSize[0]);
          let tempj = parseInt((j - this.tileSize[1] / 2) / this.tileSize[1]);
          let temp1 = parseInt(tempj * this.tilesX + tempi);
          let temp2 = parseInt(temp1 + 1);
          let temp3 = temp1 + this.tilesX;
          let temp4 = temp2 + this.tilesX;
          let p =
            (i - (tempi * this.tileSize[0] + this.tileSize[0] / 2)) /
            (1.0 * this.tileSize[0]);
          let q =
            (j - (tempj * this.tileSize[1] + this.tileSize[1] / 2)) /
            (1.0 * this.tileSize[1]);
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(
            (p * q * lutArray.get(temp4, srcVal) +
              (1 - q) * (1 - p) * lutArray.get(temp1, srcVal) +
              p * (1 - q) * lutArray.get(temp2, srcVal) +
              q * (1 - p) * lutArray.get(temp3, srcVal)) *
              255
          );
          dstArray.set(j, i, finalVal);
        }
      }
    }
    return dstArray;
  }
}
