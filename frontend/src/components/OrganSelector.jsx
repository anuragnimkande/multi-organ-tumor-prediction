import React from 'react';
import OrganCard from './OrganCard';
import { getOrganList } from '../config/organs';

const OrganSelector = ({ selectedOrgan, onSelect }) => {
  const organs = getOrganList();

  return (
    <div className="w-full max-w-5xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {organs.map((organ) => (
          <OrganCard 
            key={organ.id} 
            organ={organ} 
            isSelected={selectedOrgan?.id === organ.id}
            onClick={onSelect}
          />
        ))}
      </div>
    </div>
  );
};

export default OrganSelector;
