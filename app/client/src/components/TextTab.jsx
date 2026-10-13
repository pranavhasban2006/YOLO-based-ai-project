import { MessageSquareText } from 'lucide-react';
import { EmptyState, PageHeader } from './ui.jsx';

// Text emotion is not built yet. This page says so; there is no fake output.
export default function TextTab() {
  return (
    <div>
      <PageHeader title="Text emotion" subtitle="Reading emotion from written text." />
      <div className="card">
        <EmptyState
          icon={MessageSquareText}
          title="Not trained yet"
          text="A text emotion model is planned. For now, the Check-in chat analyses what you write during a check-in, using a language model, and shows only a summary."
        />
      </div>
    </div>
  );
}
