import React, { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import { AuthModal } from './components/AuthModal';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { RightCitationPanel } from './components/RightCitationPanel';
import { AdminDashboard } from './components/AdminDashboard';
import { ChatMessage, LegalSource } from './types';
import {
  sendChatMessageApi,
  subscribeTraceStream,
  fetchUserHistoryApi,
  deleteUserHistoryApi,
  pollTaskResultApi,
} from './services/api';

// Generate or retrieve per-browser session UUID (guest memory scope)
function getSessionUserId(): string {
  let stored = localStorage.getItem('legal_rag_session_uuid');
  if (!stored) {
    stored = 'usr_' + Math.random().toString(36).substring(2, 11);
    localStorage.setItem('legal_rag_session_uuid', stored);
  }
  return stored;
}

const AppContent: React.FC = () => {
  const { user, isAuthenticated } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const [selectedVariant, setSelectedVariant] = useState<string>('gemma4:31b-cloud');
  const [activeTab, setActiveTab] = useState<'chat' | 'admin'>('chat');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);

  // Active user ID for scoped memory and JWT association
  const activeUserId = user ? user.id : getSessionUserId();

  // Admin route guard: only authenticated admin/lawyer may view admin
  const canAccessAdmin = isAuthenticated && (user?.role === 'admin' || user?.role === 'lawyer');

  const [isRightPanelOpen, setIsRightPanelOpen] = useState<boolean>(false);
  const [selectedPanelSource, setSelectedPanelSource] = useState<LegalSource | null>(null);

  const latestAssistantMsg = [...messages].reverse().find((m) => m.role === 'assistant');
  const activeSources = latestAssistantMsg?.sources || [];

  // Force chat if user loses access to admin (e.g. logout / role change)
  useEffect(() => {
    if (activeTab === 'admin' && !canAccessAdmin) {
      setActiveTab('chat');
    }
  }, [activeTab, canAccessAdmin]);

  const handleSelectHistoryItem = (item: any) => {
    const itemId = item.conversation_id || item.id;
    setSelectedHistoryId(itemId);

    if (item.turns && Array.isArray(item.turns) && item.turns.length > 0) {
      const formatted: ChatMessage[] = [];
      item.turns.forEach((t: any, idx: number) => {
        if (t.is_request) {
          formatted.push({
            id: `h_req_${itemId}_${idx}`,
            role: 'user',
            content: t.message,
            timestamp: t.created_at || new Date().toISOString(),
          });
        } else {
          formatted.push({
            id: `h_res_${itemId}_${idx}`,
            role: 'assistant',
            content: t.message,
            timestamp: t.created_at || new Date().toISOString(),
            sources: t.sources || [],
          });
        }
      });
      setMessages(formatted);
    } else {
      const q = item.question || item.message?.question || item.content;
      const a = item.response || item.message?.response;
      if (q && a) {
        setMessages([
          {
            id: 'h_user_' + itemId,
            role: 'user',
            content: q,
            timestamp: new Date().toISOString(),
          },
          {
            id: 'h_asst_' + itemId,
            role: 'assistant',
            content: a,
            timestamp: new Date().toISOString(),
            sources: item.sources || [],
          },
        ]);
      }
    }
  };

  // Fetch conversation history on initial load or user change
  useEffect(() => {
    async function loadHistory() {
      const historyData = await fetchUserHistoryApi(activeUserId);
      setHistory(historyData);
      if (historyData.length > 0 && !selectedHistoryId) {
        handleSelectHistoryItem(historyData[0]);
      }
    }
    loadHistory();
  }, [activeUserId]);

  const handleNewChat = () => {
    setSelectedHistoryId(null);
    setMessages([]);
  };

  const handleWipeHistory = async () => {
    if (window.confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện và trí nhớ dài hạn không?')) {
      await deleteUserHistoryApi(activeUserId);
      setSelectedHistoryId(null);
      setMessages([]);
      setHistory([]);
    }
  };

  const handleSendMessage = async (text: string) => {
    // Hard gate: must be logged in to chat
    if (!isAuthenticated) return;

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

    const targetConvId = selectedHistoryId || ('sess_' + Date.now());
    if (!selectedHistoryId) {
      setSelectedHistoryId(targetConvId);
    }

    try {
      const res = await sendChatMessageApi(text, activeUserId, selectedVariant, false, targetConvId);

      if (res.task_id) {
        const taskId = res.task_id;
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

                let updatedContent = msg.content;
                if (step.payload && typeof step.payload === 'object') {
                  const ans =
                    step.payload.answer ||
                    step.payload.content ||
                    step.payload.response ||
                    step.payload.final_response ||
                    step.payload.text;
                  if (ans && typeof ans === 'string') {
                    updatedContent = ans;
                  }
                } else if (typeof step.payload === 'string' && step.payload.trim()) {
                  updatedContent = step.payload;
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
            let polledResult: any = null;
            try {
              polledResult = await pollTaskResultApi(taskId);
            } catch (e) {
              console.warn('Poll task result failed', e);
            }

            const taskRes = polledResult?.task_result || {};
            const finalContent =
              taskRes.content || taskRes.response || taskRes.answer || taskRes.text;
            const finalSources = taskRes.sources || [];
            const finalRoute = taskRes.route || '';

            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id !== assistantMsgId) return msg;
                const resolvedContent =
                  msg.content || finalContent || '⚠️ Không nhận được câu trả lời từ máy chủ.';
                return {
                  ...msg,
                  content: resolvedContent,
                  sources: msg.sources?.length ? msg.sources : finalSources,
                  route: msg.route || finalRoute,
                  isStreaming: false,
                };
              })
            );
            setIsLoading(false);
            const historyData = await fetchUserHistoryApi(activeUserId);
            setHistory(historyData);
          },
          async (err) => {
            console.error(err);
            try {
              const polledResult = await pollTaskResultApi(taskId);
              const taskRes = polledResult?.task_result || {};
              const finalContent = taskRes.content || taskRes.response || taskRes.answer;
              if (finalContent) {
                setMessages((prev) =>
                  prev.map((msg) => {
                    if (msg.id !== assistantMsgId) return msg;
                    return {
                      ...msg,
                      content: finalContent,
                      sources: taskRes.sources || [],
                      route: taskRes.route || '',
                      isStreaming: false,
                    };
                  })
                );
              }
            } catch (e) {
              /* ignore */
            }
            setIsLoading(false);
          }
        );
      } else if (res.response) {
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
    <div className="flex h-screen w-screen overflow-hidden bg-paper text-ink font-sans">
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
        selectedHistoryId={selectedHistoryId}
      />

      <div className="flex-1 flex flex-col h-full lg:pl-72 overflow-hidden">
        <Header
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          selectedVariant={selectedVariant}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
        />

        <main className="flex-1 overflow-hidden relative flex h-full">
          {activeTab === 'admin' && canAccessAdmin ? (
            <div className="h-full overflow-y-auto w-full">
              <AdminDashboard />
            </div>
          ) : (
            <>
              <div className="flex-1 h-full overflow-hidden">
                <ChatArea
                  messages={messages}
                  onSendMessage={handleSendMessage}
                  isLoading={isLoading}
                  userId={activeUserId}
                  isAuthenticated={isAuthenticated}
                  onSelectSource={(src) => {
                    setSelectedPanelSource(src);
                    setIsRightPanelOpen(true);
                  }}
                />
              </div>
              <RightCitationPanel
                sources={activeSources}
                selectedSource={selectedPanelSource}
                onSelectSource={setSelectedPanelSource}
                isOpen={isRightPanelOpen}
                onClose={() => {
                  setIsRightPanelOpen(false);
                  setSelectedPanelSource(null);
                }}
              />
            </>
          )}
        </main>
      </div>
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <ThemeProvider>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </ThemeProvider>
  );
};

export default App;