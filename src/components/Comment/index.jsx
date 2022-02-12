import "./style.less";

const Comment = ({username, comment}) => {
    return (
        <div className="comment">
            <p>{username}</p>
            <h2>{comment}</h2>
        </div>
    );
};

export default Comment;