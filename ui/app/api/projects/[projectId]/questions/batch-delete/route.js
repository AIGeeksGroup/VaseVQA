import { NextResponse } from 'next/server';
import { batchDeleteQuestions } from '@/lib/db/questions';

// 批量删除问题
export async function DELETE(request, { params }) {
  try {
    const { projectId } = params;
    
    // 验证项目ID
    if (!projectId) {
      return NextResponse.json({ error: '项目ID不能为空' }, { status: 400 });
    }
    
    // 从请求体中获取要删除的问题列表
    const { questions } = await request.json();
    
    if (!questions || !Array.isArray(questions) || questions.length === 0) {
      return NextResponse.json({ error: '问题列表不能为空' }, { status: 400 });
    }
    
    // 验证每个问题都有必要的字段
    for (const question of questions) {
      if (!question.questionId || !question.chunkId) {
        return NextResponse.json({ error: '问题信息不完整，需要包含 questionId 和 chunkId' }, { status: 400 });
      }
    }
    
    // 批量删除问题
    await batchDeleteQuestions(projectId, questions);
    
    return NextResponse.json({ 
      success: true, 
      message: `成功删除 ${questions.length} 个问题` 
    });
  } catch (error) {
    console.error('批量删除问题失败:', error);
    return NextResponse.json({ error: error.message || '批量删除问题失败' }, { status: 500 });
  }
}
