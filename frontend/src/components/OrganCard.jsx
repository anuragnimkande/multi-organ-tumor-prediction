import React from 'react';
import StatusBadge from './StatusBadge';

const OrganCard = ({ organ, onClick, isSelected }) => {
  const { name, modality, icon, color, status, description } = organ;
  const disabled = status !== 'active';

  return (
    <div 
      onClick={() => !disabled && onClick(organ)}
      className={`glass-card p-6 relative overflow-hidden transition-all duration-300
        ${disabled ? 'opacity-60 grayscale cursor-not-allowed' : 'cursor-pointer hover:-translate-y-1'}
        ${isSelected ? 'ring-2 ring-offset-2 ring-offset-bg-base' : 'hover:border-quantum-500/50'}
      `}
      style={{
        '--tw-ring-color': color,
        boxShadow: isSelected ? `0 0 20px ${color}40` : '',
      }}
    >
      {/* Decorative gradient blob */}
      <div 
        className="absolute -top-12 -right-12 w-32 h-32 rounded-full blur-3xl opacity-20"
        style={{ backgroundColor: color }}
      ></div>

      <div className="flex justify-between items-start mb-4">
        <div 
          className="text-4xl p-3 rounded-xl bg-white/5 border border-white/10"
          style={{ boxShadow: `inset 0 0 10px ${color}20` }}
        >
          {icon}
        </div>
        <StatusBadge status={status} />
      </div>

      <h3 className="text-xl font-bold text-white mb-1">{name}</h3>
      <p className="text-sm font-medium mb-3" style={{ color }}>{modality}</p>
      
      <p className="text-sm text-slate-400 line-clamp-2">
        {description}
      </p>

      {isSelected && (
        <div className="absolute inset-x-0 bottom-0 h-1" style={{ backgroundColor: color }}></div>
      )}
    </div>
  );
};

export default OrganCard;
