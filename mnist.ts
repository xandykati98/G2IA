import { NeuralNetwork, scaleBetween } from './nn_organized';
// 10000 train, 2500 test
import train_set from './mnist/train.json';
import test_set from './mnist/test.json';
import { SemiBrief } from './descriptors/semibrief';
import { benchmark } from './benchmark';

const train = train_set as { data: number[]; label: string }[];
const test = test_set as { data: number[]; label: string }[];

const rede = new NeuralNetwork()

const descriptor = new SemiBrief(33, 28**2, 28, 28)

rede.pushLayer({
    is_input: true,
    neurons_number: 33
})

rede.pushLayer({
    neurons_number: 65,
})
rede.pushLayer({
    neurons_number: 32,
})

rede.pushLayer({
    is_output: true,
    neurons_number: 10,
})

rede.createWeights()
const label_to_output = (label: string) => {
    const output = new Array(10).fill(0)
    output[parseInt(label)] = 1
    return output
}
const normalized_train_set = train.map(item => ({ 
    inputs: descriptor.encode(item.data.map(num => num/255)), 
    desired_outputs: label_to_output(item.label) 
}))
const normalized_test_set = test.map(item => ({
    inputs: descriptor.encode(item.data.map(num => num/255)),
    desired_outputs: label_to_output(item.label)
}))

const train_config = {
    taxa_aprendizado: 0.05,
    epochs: 30,
    iteracoes: 3200,
    training_set: normalized_train_set,
    momentum: 0.25,
}

benchmark({
    descriptor: descriptor,
    v_set: normalized_test_set,
    train_config: train_config,
    rede: rede,
    runs:1,
    on_run_end: () => {
        descriptor.reset()
    },
    get_prediction_from_array: output => {
        const max = Math.max(...output)
        
        return [...output].map((num) => num === max ? 1 : 0)
    },
    bechnmark_name: "mnist-33",
})

/**
let v_i = 0;

let acertos = 0
while (v_i < test.length) {
    const test_item = test[v_i]
    const test_item_inputs = test_item.data.map(num => scaleBetween(num, 0, 1, 0, 255))
    const test_item_outputs = rede.guess(test_item_inputs)
    const test_item_label = test_item.label
    const test_item_label_output = label_to_output(test_item_label)

    if (test_item_label_output.toString() === test_item_outputs.toString()) {
        acertos++
    }
    console.log(`Acertos: ${acertos} de ${test.length}`)
    v_i++
}
console.log(`Acertos: ${acertos} de ${test.length}`) */