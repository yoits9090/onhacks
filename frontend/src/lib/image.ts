/* Image pipeline helpers. Direct port of demo/exposure.js —
 * header sniffing before decode (giant-allocation guard), EXIF-safe decode,
 * canvas construction and PNG export. */

export const MAX_BYTES = 10 * 1024 * 1024
export const MAX_PIXELS = 50_000_000
export const MAX_EDGE = 1600
export const MAX_EDITS = 12

/* Exact cover color used in verified rehearsals (RGB 21,33,57). */
export const COVER_FILL = '#152139'

export interface ImageHeader {
  type: 'image/png' | 'image/webp' | 'image/jpeg'
  width: number
  height: number
}

// Read dimensions before decoding: prevent giant allocations from small
// compressed files. This inspects only container headers, not EXIF.
export function inspectImageHeader(buffer: ArrayBuffer): ImageHeader {
  const b = new Uint8Array(buffer)
  const v = new DataView(buffer)
  const text = (start: number, length: number) =>
    String.fromCharCode(...b.subarray(start, start + length))
  if (b.length >= 24 && b[0] === 137 && text(1, 3) === 'PNG' && text(12, 4) === 'IHDR') {
    return { type: 'image/png', width: v.getUint32(16), height: v.getUint32(20) }
  }
  if (b.length >= 12 && text(0, 4) === 'RIFF' && text(8, 4) === 'WEBP') {
    for (let at = 12; at + 8 <= b.length; ) {
      const name = text(at, 4)
      const size = v.getUint32(at + 4, true)
      const data = at + 8
      if (data + size > b.length) break
      const uint24 = (pos: number) => b[pos] | (b[pos + 1] << 8) | (b[pos + 2] << 16)
      if (name === 'VP8X' && size >= 10)
        return {
          type: 'image/webp',
          width: uint24(data + 4) + 1,
          height: uint24(data + 7) + 1,
        }
      if (name === 'VP8L' && size >= 5 && b[data] === 0x2f) {
        const bits = v.getUint32(data + 1, true)
        return {
          type: 'image/webp',
          width: (bits & 0x3fff) + 1,
          height: ((bits >>> 14) & 0x3fff) + 1,
        }
      }
      if (name === 'VP8 ' && size >= 10 && b[data + 3] === 0x9d && b[data + 4] === 1 && b[data + 5] === 0x2a) {
        return {
          type: 'image/webp',
          width: v.getUint16(data + 6, true) & 0x3fff,
          height: v.getUint16(data + 8, true) & 0x3fff,
        }
      }
      at = data + size + (size % 2)
    }
  }
  if (b.length > 4 && b[0] === 0xff && b[1] === 0xd8) {
    const sof = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf])
    let at = 2
    while (at + 3 < b.length) {
      if (b[at++] !== 0xff) break
      while (b[at] === 0xff) at++
      const marker = b[at++]
      if (marker === 0xda || marker === 0xd9) break
      if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue
      if (at + 2 > b.length) break
      const size = v.getUint16(at)
      if (size < 2 || at + size > b.length) break
      if (sof.has(marker) && size >= 8)
        return { type: 'image/jpeg', width: v.getUint16(at + 5), height: v.getUint16(at + 3) }
      at += size
    }
  }
  throw new Error('This file is not a readable JPEG, PNG or WebP. Choose a supported still image.')
}

export async function decodeImage(blob: Blob): Promise<ImageBitmap | HTMLImageElement> {
  if ('createImageBitmap' in window) {
    // The browser applies EXIF orientation exactly once. Do not rotate again.
    try {
      return await createImageBitmap(blob, { imageOrientation: 'from-image' })
    } catch {
      /* Older browser codecs may support HTMLImageElement instead. */
    }
  }
  const url = URL.createObjectURL(blob)
  try {
    const image = new Image()
    image.src = url
    await image.decode()
    return image
  } finally {
    URL.revokeObjectURL(url)
  }
}

export function makeCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  return canvas
}

export function exportPNG(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob(blob => {
      if (!blob)
        reject(new Error('The browser could not export this image. Try a smaller photo.'))
      else resolve(blob)
    }, 'image/png'),
  )
}

export function sameOriginImage(url: unknown): string {
  if (typeof url !== 'string')
    throw new Error('The recorded image is missing. Choose another example or your own photo.')
  const parsed = new URL(url, window.location.href)
  if (parsed.origin !== window.location.origin || !['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error('The example image must be served by this demo.')
  }
  return parsed.href
}
