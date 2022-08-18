
// Ref.: https://github.com/tensorflow/playground/blob/02469bd3751764b20486015d4202b792af5362a6/src/dataset.ts

import { benchmark } from "./benchmark";
import { NeuralNetwork, scaleBetween } from "./nn_organized";

type Example2D = {
    x: number,
    y: number,
    label: number
}
let n = 100;
const noise = 0.0001;

/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a: number, b: number) {
    return Math.random() * (b - a) + a;
}
function genSpiral(deltaT: number, label: number) {
    let points: Example2D[] = [];
    for (let i = 0; i < n; i++) {
        let r = i / n * 8;
        let t = 1.75 * i / n * 2 * Math.PI + deltaT;
        let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
        let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
        points.push({x, y, label});
    }
    return points
}

const po = genSpiral(0, 1); // Positive examples.
const ne = genSpiral(Math.PI, -1); // Negative examples.


let rede = new NeuralNetwork()
rede.pushLayer({
    is_input: true,
    neurons_number: 7,
})

rede.pushLayer({
    neurons_number:12,
    inline_bias: true,
    activation_function: 'relu'
})
rede.pushLayer({
    neurons_number:8,
    inline_bias: true,
    activation_function: 'relu'
})

rede.pushLayer({
    neurons_number:8,
    inline_bias: true,
    activation_function: 'sigmoid'
})

rede.pushLayer({
    is_output: true,
    neurons_number: 1,
})

rede.initAllWeights()

const train_set = [...po, ...ne].map(sample => {
    return {
        inputs: [
            sample.x, // scaleBetween(sample.x, -1, 1, -8, 6),
            sample.y, // scaleBetween(sample.y, -1, 1, -8, 5),
            sample.x*sample.y, // scaleBetween(sample.x*sample.y, -1, 1, -(8*8), 5*6),
            // // weird
            sample.x**2, // scaleBetween(sample.x**2, -1, 1, 0, 6**2), 
            // // weird
            sample.y**2, // scaleBetween(sample.y**2, -1, 1, 0, 5**2),
            Math.sin(sample.x), 
            Math.sin(sample.y)
        ],
        desired_outputs: [sample.label],
    }
});
const input_mins = train_set.reduce((acc, sample) => {
    return {
        x: Math.min(acc.x, sample.inputs[0]),
        y: Math.min(acc.y, sample.inputs[1]),
        xy: Math.min(acc.xy, sample.inputs[2]),
        x2: Math.min(acc.x2, sample.inputs[3]),
        y2: Math.min(acc.y2, sample.inputs[4]),
        sinx: Math.min(acc.sinx, sample.inputs[5]),
        siny: Math.min(acc.siny, sample.inputs[6]),
    }
}, {
    x: Infinity,
    y: Infinity,
    xy: Infinity,
    x2: Infinity,
    y2: Infinity,
    sinx: Infinity,
    siny: Infinity,
});
const input_maxs = train_set.reduce((acc, sample) => {
    return {
        x: Math.max(acc.x, sample.inputs[0]),
        y: Math.max(acc.y, sample.inputs[1]),
        xy: Math.max(acc.xy, sample.inputs[2]),
        x2: Math.max(acc.x2, sample.inputs[3]),
        y2: Math.max(acc.y2, sample.inputs[4]),
        sinx: Math.max(acc.sinx, sample.inputs[5]),
        siny: Math.max(acc.siny, sample.inputs[6]),
    }
}, {
    x: -Infinity,
    y: -Infinity,
    xy: -Infinity,
    x2: -Infinity,
    y2: -Infinity,
    sinx: -Infinity,
    siny: -Infinity,
});

const scaled_train_set = train_set.map(sample => {
    return {
        inputs: [
            scaleBetween(sample.inputs[0], -1, 1, input_mins.x, input_maxs.x),
            scaleBetween(sample.inputs[1], -1, 1, input_mins.y, input_maxs.y),
            scaleBetween(sample.inputs[2], -1, 1, input_mins.xy, input_maxs.xy),
            scaleBetween(sample.inputs[3], -1, 1, input_mins.x2, input_maxs.x2),
            scaleBetween(sample.inputs[4], -1, 1, input_mins.y2, input_maxs.y2),
            scaleBetween(sample.inputs[5], -1, 1, input_mins.sinx, input_maxs.sinx),
            scaleBetween(sample.inputs[6], -1, 1, input_mins.siny, input_maxs.siny),
        ],
        desired_outputs: [sample.desired_outputs[0]],
    }
});

const train_config = {
    epochs: 15,
    momentum: 0.01,
    iteracoes: 10000,
    taxa_aprendizado: 0.04,
    training_set: train_set,
}

benchmark({
    v_set: train_set,
    t_set: train_set,
    train_config: train_config,
    rede: rede,
    runs: 1,
    get_prediction: output => Math.round(output) ? 1 : -1,
    bechnmark_name: "spiral",
    saveModel: true,
})