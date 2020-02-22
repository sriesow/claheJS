let nj = require('numjs');
nj.config.printThreshold = 300;

export class CLAHE {
  constructor(limit, tilesX, tilesY) {
    this.tilesX = tilesX;
    this.tilesY = tilesY;
    this.limit = limit;
    this.histSize = 256;
    this.tileSize = [];
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
            tileHist.set(numBlock, i, histValue+redistBatch);
          }
          if(residual!=0){
            let residualStep = Math.max(this.histSize/residual, 1);
            let i = 0;
            while (i<this.histSize && residual>0) {
              let histValue = tileHist.get(numBlock, i);
              tileHist.set(numBlock, i, histValue+1);
              i += residualStep;
              residual--;
            }
          }
        }

        // Calculate lut
        lutArray.set(numBlock, i, tileHist.get(numBlock, 0)/tileSizeTotal);
        for (let i = 1; i < this.histSize; i++) {
          let tempVal = lutArray.get(numBlock, i-1) + tileHist.get(numBlock, i)/tileSizeTotal;
          lutArray.set(numBlock, i, tempVal);
        }
        // console.log(tileHist.pick(numBlock, null), lutArray.pick(numBlock, null));
        // console.log(lutArray.pick(numBlock, null));
      }
    }
    return lutArray;
  }

  calculateInterpolationBody(srcArray, lutArray, height, width){
    let dstArray = srcArray.clone();
    for (let i = 0; i < width; i++) {
      for (let j = 0; j < height; j++) {
        if(i <= this.tileSize[0]/2 && j<= this.tileSize[1]/2){
          let tempVal = 0;
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(lutArray.get(tempVal,srcVal) * 255);
          dstArray.set(j,i, finalVal);
        }
        else if(i<= this.tileSize[0]/2 && j >= ((this.tilesX-1)*this.tileSize[1] + this.tileSize[1]/2)){
          let tempVal = this.tilesX*(this.tilesX-1);
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(lutArray.get(tempVal,srcVal) * 255);
          dstArray.set(j,i, finalVal);
        }
        else if(i >= ((this.tilesX-1)*this.tileSize[0] + this.tileSize[0]/2) && j<= this.tileSize[1]/2){
          let tempVal = this.tilesX-1;
          let srcVal = dstArray.get(j, i);
          let finalVal = parseInt(lutArray.get(tempVal,srcVal) * 255);
          dstArray.set(j,i, finalVal);
        }
        else if(i>=() && j>=())
      }
      
    }
  }

  CLAHE_IMPL(inputArray, height, width) {
    let lutArray = this.calculateLUTBody(inputArray, height, width);
    let outputArray = this.calculateInterpolationBody(inputArray, lutArray, height, width);
  }

}

let claheObj = new CLAHE(4.0, 2, 2);
let inputArray = [
  72,
  136,
  128,
  59,
  223,
  85,
  252,
  75,
  246,
  84,
  111,
  107,
  249,
  245,
  229,
  94
];
let njArray = nj.array(inputArray).reshape(4, 4);
claheObj.CLAHE_IMPL(njArray, 4, 4);
