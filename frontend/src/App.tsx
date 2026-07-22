import React, { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { AuthModal } from './components/AuthModal';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { AdminDashboard } from './components/AdminDashboard';
import { ChatMessage } from './types';
import {
  sendChatMessageApi,
  subscribeTraceStream,
  fetchUserHistoryApi,
  deleteUserHistoryApi,
} from './services/api';

// Generate or retrieve per-browser session UUID
function getSessionUserId(): string {
  let stored = localStorage.getItem('legal_rag_session_uuid');
  if (!stored) {
    stored = 'usr_' + Math.random().toString(36).substring(2, 11);
    localStorage.setItem('legal_rag_session_uuid', stored);
  }
  return stored;
}

const AppContent: React.FC = () => {
  const { user } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const [selectedVariant, setSelectedVariant] = useState<string>('gemma4:31b-cloud');
  const [activeTab, setActiveTab] = useState<'chat' | 'admin'>('chat');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Active user ID for scoped memory and JWT association
  const activeUserId = user ? user.id : getSessionUserId();

  // Fetch conversation history on initial load or user change
  useEffect(() => {
    async function loadHistory() {
      const historyData = await fetchUserHistoryApi(activeUserId);
      setHistory(historyData);
    }
    loadHistory();
  }, [activeUserId]);

  const handleNewChat = () => {
    setMessages([]);
  };

  const handleSelectHistoryItem = (item: any) => {
    if (item.question && item.response) {
      setMessages([
        {
          id: 'h_user_' + item.id,
          role: 'user',
          content: item.question,
          timestamp: new Date().toISOString(),
        },
        {
          id: 'h_asst_' + item.id,
          role: 'assistant',
          content: item.response,
          timestamp: new Date().toISOString(),
          sources: item.sources || [],
        },
      ]);
    }
  };

  const handleWipeHistory = async () => {
    if (window.confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện và trí nhớ dài hạn không?')) {
      await deleteUserHistoryApi(activeUserId);
      setMessages([]);
      setHistory([]);
    }
  };

  const handleSendMessage = async (text: string) => {
    const userMsgId = 'u_' + Date.now();
    const assistantMsgId = 'a_' + Date.now();

    const newMessages: ChatMessage[] = [
      ...messages,
      {
        id: userMsgId,
        role: 'user',
        content: text,
        timestamp: new Date().toISOString(),
      },
      {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isStreaming: true,
        traceSteps: [],
      },
    ];

    setMessages(newMessages);
    setIsLoading(true);

    try {
      // Dispatch chat message request to backend
      const res = await sendChatMessageApi(text, activeUserId, selectedVariant, false);

      if (res.task_id) {
        const taskId = res.task_id;

        // Subscribe to SSE stream for real-time trace events
        const unsubscribe = subscribeTraceStream(
          taskId,
          (step: any) => {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id !== assistantMsgId) return msg;

                const newTraceSteps = [...(msg.traceSteps || [])];
                if (step.event_type && step.node) {
                  newTraceSteps.push({
                    node: step.node,
                    step_index: step.step_index,
                    event_type: step.event_type,
                    payload: step.payload,
                  });
                }

                // Final answer step payload update
                let updatedContent = msg.content;
                if (step.payload && typeof step.payload === 'object' && step.payload.answer) {
                  updatedContent = step.payload.answer;
                }

                return {
                  ...msg,
                  content: updatedContent,
                  traceSteps: newTraceSteps,
                };
              })
            );
          },
          async () => {
            // Finished streaming: poll final result or mark complete
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id !== assistantMsgId) return msg;
                return { ...msg, isStreaming: false };
              })
            );
            setIsLoading(false);
            // Refresh history
            const historyData = await fetchUserHistoryApi(activeUserId);
            setHistory(historyData);
          },
          (err) => {
            console.error(err);
            setIsLoading(false);
          }
        );
      } else if (res.response) {
        // Sync response fallback
        setMessages((prev) =>
          prev.map((msg) => {
            if (msg.id !== assistantMsgId) return msg;
            return {
              ...msg,
              content: res.response || '',
              sources: res.sources || [],
              route: res.route,
              tool_calls: res.tool_calls,
              isStreaming: false,
            };
          })
        );
        setIsLoading(false);
        const historyData = await fetchUserHistoryApi(activeUserId);
        setHistory(historyData);
      }
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id !== assistantMsgId) return msg;
          return {
            ...msg,
            content: `⚠️ Đã xảy ra lỗi: ${err.message || 'Không thể kết nối đến máy chủ backend.'}`,
            isStreaming: false,
          };
        })
      );
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
      <AuthModal />

      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onNewChat={handleNewChat}
        history={history}
        onSelectHistoryItem={handleSelectHistoryItem}
        onWipeHistory={handleWipeHistory}
        selectedVariant={selectedVariant}
        onSelectVariant={setSelectedVariant}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      <div className="flex-1 flex flex-col h-full lg:pl-72 overflow-hidden">
        <Header
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          selectedVariant={selectedVariant}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
        />

        <main className="flex-1 overflow-hidden relative">
          {activeTab === 'chat' ? (
            <ChatArea
              messages={messages}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              userId={activeUserId}
            />
          ) : (
            <div className="h-full overflow-y-auto">
              <AdminDashboard />
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;
