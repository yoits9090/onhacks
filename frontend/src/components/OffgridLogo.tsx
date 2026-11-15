import { useReducedMotion } from 'framer-motion'
import MetallicPaint from '@/components/metallic/MetallicPaint'
import logo from '@/components/metallic/offgrid-mark.svg'

export default function OffgridLogo() {
  const reduceMotion = useReducedMotion()
  return (
    <a href="/" className="split-brand" aria-label="Offgrid home">
      <span className="metallic-logo" aria-hidden="true">
        <img src={logo} alt="" />
        {!reduceMotion && <MetallicPaint imageSrc={logo} density={2.6} dispersion={0.45}
          distortion={0.22} edgeBend={0.92} scale={1} intro={false} mouseTilt={0} />}
      </span>
      <span className="brand-name">offgrid.</span>
    </a>
  )
}
