import { Descriptor } from "./descriptor";

/**
 * Acha pares de forma aleatória
 */
export class SemiBrief extends Descriptor {
    pairs_qtd: number;
    pairs: number[][]
    constructor(pairs_qtd:number, flattened_image_size: number, original_w: number, original_h: number) {
        super(flattened_image_size, original_w, original_h)
        this.pairs_qtd = pairs_qtd
        this.pairs = []
        
        this.reset()
    }
    randomIntFromInterval(min: number, max: number) {
        return Math.floor(Math.random() * (max - min + 1) + min);
    }
    reset() {
        this.pairs = []
        
        for (let i = 0; i < this.pairs_qtd; i++) {
            this.pairs.push([this.randomIntFromInterval(0, this.flattened_image_size), this.randomIntFromInterval(0, this.flattened_image_size)])
        }
    }
    encode(flattened_image:number[]): number[] {
        const encoded:number[] = []
        for (const [ p1, p2 ] of this.pairs) {
            encoded.push(flattened_image[p1] > flattened_image[p2] ? 1 : -1)
        }
        return encoded
    }
}