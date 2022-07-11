// Ref.: https://gist.github.com/venil7/534dfedcff23b0e8fbb5626e7d169677
const fs = require('fs');
const dataFileBuffer = fs.readFileSync(__dirname + '/train-images-idx3-ubyte');
const labelFileBuffer = fs.readFileSync(__dirname+'/train-labels-idx1-ubyte');
const pixelValues     = [];

for (let image = 0; image <= 59999; image++) {
    const pixels = [];

    for (let x = 0; x <= 27; x++) {
        for (let y = 0; y <= 27; y++) {
            pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15]);
        }
    }

    const label = JSON.stringify(labelFileBuffer[image + 8]);
    const data = pixels;
    const imageData = {
        label,
        data
    };
    if (pixelValues.length === 10000) break;
    pixelValues.push(imageData);
}
fs.writeFileSync(__dirname + '/train.json', JSON.stringify(pixelValues));