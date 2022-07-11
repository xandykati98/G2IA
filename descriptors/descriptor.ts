export class Descriptor {
    flattened_image_size: number;
    original_w: number;
    original_h: number;
    name:string;
    constructor(flattened_image_size: number, original_w: number, original_h: number) {
        this.flattened_image_size = flattened_image_size;
        this.original_w = original_w;
        this.original_h = original_h;
        this.name = this.constructor.name;
    }
    encode(..._):any {
    }
    reset(..._):any {
    }
}